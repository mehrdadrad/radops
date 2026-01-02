import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import List

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from integrations.github.github_loader import GithubLoader
from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class PineconeVectorStoreManager:
    """
    Manages a Pinecone vector store, ensuring it's synchronized with a directory of text files.
    """

    def __init__(
        self,
        name: str,
        sync_locations: list[SyncLocationSettings],
        embeddings,
        sync_interval: int = 60,
        skip_initial_sync: bool = False
    ):
        """
        Initializes the manager, connects to Pinecone, and synchronizes the vector store.

        Args:
            name: The name of the vector store manager instance.
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
            skip_initial_sync: If True, skips the initial data synchronization.
        """
        self._config = settings.vector_store.providers["pinecone"]
        self._api_key = self._config.get("api_key")
        self._index_name = self._config.get("index_name")
        self._closed = False

        if not self._api_key or not self._index_name:
            raise ValueError(
                "Pinecone 'api_key' and 'index_name' are required in "
                "vector_store.providers.pinecone config."
            )

        try:
            self._pc = Pinecone(api_key=self._api_key)
            # Ensure index exists or access it.
            # Note: Creating an index is a heavy operation, we assume it exists
            # or user creates it.
            self._index = self._pc.Index(self._index_name)
            logger.info(
                "Successfully connected to Pinecone Index: %s", self._index_name
            )
        except Exception as e:
            logger.error("Failed to connect to Pinecone: %s", e, exc_info=True)
            raise

        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._name = name
        self._stop_event = threading.Event()
        self._vectorstores = {}
        self._loaders = []

        for location in self._sync_locations:
            # We use the 'collection' name as the Pinecone 'namespace'
            namespace = location.collection

            poll_interval = (
                location.sync_interval
                if location.sync_interval is not None
                else self._sync_interval
            )
            if location.type == "fs":
                loader = FileSystemLoader(
                    path=location.path,
                    poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": namespace,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[namespace] = PineconeVectorStore(
                    index=self._index,
                    embedding=embeddings,
                    namespace=namespace
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path],
                    poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": namespace,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[namespace] = PineconeVectorStore(
                    index=self._index,
                    embedding=embeddings,
                    namespace=namespace
                )
            elif location.type == "github":
                loader = GithubLoader(
                    repo_names=location.path.split(","),
                    loader_config=location.loader_config,
                    poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": namespace,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[namespace] = PineconeVectorStore(
                    index=self._index,
                    embedding=embeddings,
                    namespace=namespace
                )
            else:
                logger.warning("Unsupported sync location type: %s", location.type)

        if not skip_initial_sync:
            self._initial_sync()

        if self._sync_interval > 0:
            self.start_periodic_sync()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context, ensuring resources are closed."""
        self.close()

    def _initial_sync(self):
        """
        Performs an initial data synchronization for all configured locations.
        """
        logger.info("Performing initial data synchronization for all locations...")
        for loader_info in self._loaders:
            loader = loader_info["loader"]
            collection_name = loader_info["collection"]
            logger.info("Loading initial data for namespace '%s'...", collection_name)
            self._update_vector_store(loader.load_data(), collection_name)

    def _update_vector_store(
        self, changed_docs: List[LoadedDocument], collection: str
    ):
        """
        Updates the Pinecone namespace with changed documents.
        """
        vectorstore = self._vectorstores.get(collection)
        if not vectorstore:
            logger.error("No vector store found for namespace '%s'.", collection)
            return

        location_config = next(
            (loc for loc in self._sync_locations
             if loc.collection == collection),
            None
        )

        # Get dimension for dummy vector
        dimension = 1536
        try:
            stats = self._index.describe_index_stats()
            dimension = (
                stats.get("dimension", 1536)
                if isinstance(stats, dict)
                else getattr(stats, "dimension", 1536)
            )
        except Exception as e:
            logger.debug(f"Could not get index stats: {e}")

        docs_to_upsert = [
            doc for doc in changed_docs if doc.content is not None
        ]
        docs_to_delete = [doc for doc in changed_docs if doc.content is None]

        # Filter out documents that are already up-to-date in Pinecone
        if docs_to_upsert:
            docs_that_need_update = []
            # Use a non-zero vector to avoid errors with Cosine similarity
            dummy_vector = [0.01] * dimension
            for doc in docs_to_upsert:
                source = os.path.basename(doc.path)
                try:
                    # Check if any vector exists with the same source
                    # Query by source only, then compare timestamp manually
                    results = self._index.query(
                        vector=dummy_vector,
                        top_k=1,
                        filter={
                            "source": source
                        },
                        namespace=collection,
                        include_metadata=True
                    )
                    
                    needs_update = True
                    if results.matches:
                        metadata = results.matches[0].metadata or {}
                        stored_mod = metadata.get("last_modification")
                        
                        if stored_mod is not None:
                            # Compare as integers to handle float storage in Pinecone
                            if int(float(stored_mod)) == int(doc.last_modified):
                                needs_update = False
                                logger.debug(
                                    "Skipping '%s': up-to-date (ts: %s)", 
                                    source, int(stored_mod)
                                )
                    
                    if needs_update:
                        docs_that_need_update.append(doc)
                except Exception as e:
                    logger.warning(
                        "Error checking existence of '%s' in Pinecone: %s", source, e
                    )
                    docs_that_need_update.append(doc)
            docs_to_upsert = docs_that_need_update

        # Handle deletions
        if docs_to_delete:
            deleted_sources = [
                os.path.basename(doc.path) for doc in docs_to_delete
            ]
            logger.info(
                "Removing %d deleted files from namespace '%s': %s",
                len(deleted_sources),
                collection,
                ', '.join(sorted(deleted_sources))
            )
            # Pinecone delete with filter
            try:
                vectorstore.delete(filter={"source": {"$in": deleted_sources}})
            except Exception as e:
                if "Namespace not found" in str(e):
                    logger.debug(
                        "Namespace '%s' not found during deletion (expected on first run).",
                        collection
                    )
                else:
                    logger.warning(
                        "Error deleting from Pinecone namespace '%s': %s", collection, e
                    )

        # Handle upserts: Delete existing chunks for these files first to avoid duplicates
        if docs_to_upsert:
            upsert_sources = [
                os.path.basename(doc.path) for doc in docs_to_upsert
            ]
            logger.info(
                "Upserting %d files in namespace '%s'",
                len(upsert_sources),
                collection
            )
            try:
                vectorstore.delete(filter={"source": {"$in": upsert_sources}})
            except Exception as e:
                if "Namespace not found" in str(e):
                    logger.debug(
                        "Namespace '%s' not found during deletion (expected on first run).",
                        collection
                    )
                else:
                    logger.warning(
                        "Error deleting existing vectors in Pinecone namespace '%s': %s",
                        collection,
                        e
                    )

        # Prepare and add new/updated documents
        documents_to_add = []
        for loaded_doc in docs_to_upsert:
            file_name = os.path.basename(loaded_doc.path)
            metadata = {
                "source": file_name,
                "last_modification": loaded_doc.last_modified
            }

            try:
                if (location_config and
                        location_config.metadata and
                        location_config.metadata.structure):
                    file_name_no_ext = Path(file_name).stem
                    parts = file_name_no_ext.split(
                        location_config.metadata.delimiter
                    )
                    for i, prop_setting in enumerate(
                        location_config.metadata.structure
                    ):
                        if i < len(parts):
                            metadata[prop_setting.name] = parts[i]
            except (ValueError, AttributeError) as e:
                logger.warning(
                    "Could not extract metadata from filename '%s': %s", file_name, e
                )

            doc = Document(page_content=loaded_doc.content, metadata=metadata)
            documents_to_add.append(doc)

        if documents_to_add:
            chunk_size = 500
            chunk_overlap = 50
            if location_config and location_config.loader_config:
                chunk_size = location_config.loader_config.get("chunk_size", 500)
                chunk_overlap = location_config.loader_config.get("chunk_overlap", 50)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            docs: List[Document] = text_splitter.split_documents(
                documents_to_add
            )
            logger.info(
                "Adding %d document chunks to namespace '%s'...", len(docs), collection
            )
            vectorstore.add_documents(docs)

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        for loader_info in self._loaders:
            if loader_info['poll_interval'] == 0:
                logger.info(
                    "Skipping periodic synchronization watcher, collection: %s",
                    loader_info['collection']
                )
                continue
            update_callback = partial(
                self._update_vector_store,
                collection=loader_info['collection']
            )
            loader_info["loader"].watcher(callback=update_callback)
            logger.info(
                "Starting periodic synchronization watcher, collection: %s",
                loader_info['collection']
            )

    def stop_periodic_sync(self):
        """Stops the background synchronization thread gracefully."""
        logger.info("Stopping periodic synchronization watcher...")
        for loader_info in self._loaders:
            loader_info["loader"].stop_watcher()
            loader_info["loader"].close()

    def get_vectorstore(self, collection: str) -> PineconeVectorStore:
        """Returns the Pinecone vector store instance for a given namespace."""
        return self._vectorstores.get(collection)

    def name(self) -> str:
        return self._name

    def close(self):
        """Closes client connections and stops background threads."""
        if self._closed:
            return
        self.stop_periodic_sync()
        self._closed = True
        # Pinecone client doesn't strictly require a close, but good practice if available
        logger.info("Pinecone manager closed.")