"""
This module implements the FaissVectorStoreManager for managing FAISS vector stores.
"""
import logging
import os
from functools import partial
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from integrations.github.github_loader import GithubLoader
from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class FaissVectorStoreManager:
    """
    Manages a FAISS vector store, ensuring it's synchronized with a directory of text files.
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
        Initializes the manager, loads/creates FAISS indices, and synchronizes.

        Args:
            name: The name of the vector store manager instance.
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
            skip_initial_sync: If True, skips the initial data synchronization.
        """
        self._config = settings.vector_store.providers.get("faiss", {})
        self._persistent = self._config.get("persistent", False)
        self._index_root = Path(self._config.get("path", "faiss_indexes"))
        if self._persistent:
            self._index_root.mkdir(parents=True, exist_ok=True)

        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._name = name
        self._vectorstores = {}
        self._loaders = []

        for location in self._sync_locations:
            collection_name = location.collection
            index_path = self._index_root / collection_name

            # Try to load existing index
            if self._persistent and index_path.exists() and (index_path / "index.faiss").exists():
                try:
                    self._vectorstores[collection_name] = FAISS.load_local(
                        str(index_path),
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Loaded FAISS index for collection '%s'", collection_name)
                except Exception as e:
                    logger.error(
                        "Failed to load FAISS index for '%s': %s", collection_name, e
                    )
                    self._vectorstores[collection_name] = None
            else:
                self._vectorstores[collection_name] = None

            # Initialize loaders
            poll_interval = (
                location.sync_interval
                if location.sync_interval is not None
                else self._sync_interval
            )
            
            loader = None
            if location.type == "fs":
                loader = FileSystemLoader(
                    path=location.path, poll_interval=poll_interval
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path], poll_interval=poll_interval
                )
            elif location.type == "github":
                loader = GithubLoader(
                    repo_names=location.path.split(","),
                    loader_config=location.loader_config,
                    poll_interval=poll_interval
                )
            else:
                logger.warning("Unsupported sync location type: %s", location.type)
                continue

            if loader:
                self._loaders.append({
                    "loader": loader,
                    "collection": collection_name,
                    "poll_interval": poll_interval
                })

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
            logger.info("Loading initial data for collection '%s'...", collection_name)
            self._update_vector_store(loader.load_data(), collection_name)

    def _update_vector_store(self, changed_docs: List[LoadedDocument], collection: str):
        """
        Updates the FAISS index with changed documents.
        """
        location_config = next(
            (loc for loc in self._sync_locations if loc.collection == collection),
            None
        )
        
        docs_to_upsert = [doc for doc in changed_docs if doc.content is not None]
        docs_to_delete = [doc for doc in changed_docs if doc.content is None]
        
        vs = self._vectorstores.get(collection)

        # Handle Deletions and Updates (remove old versions)
        sources_to_remove = set()
        if docs_to_delete:
            sources_to_remove.update(os.path.basename(d.path) for d in docs_to_delete)
        if docs_to_upsert:
            sources_to_remove.update(os.path.basename(d.path) for d in docs_to_upsert)

        if vs and sources_to_remove:
            ids_to_delete = []
            # Iterate over docstore to find IDs matching source
            # FAISS wrapper uses a docstore (InMemoryDocstore by default)
            try:
                for doc_id, doc in vs.docstore._dict.items():
                    if doc.metadata.get("source") in sources_to_remove:
                        ids_to_delete.append(doc_id)
            except AttributeError:
                logger.warning("Could not access docstore to perform deletions.")
            
            if ids_to_delete:
                logger.info(
                    "Deleting %d chunks from FAISS collection '%s'",
                    len(ids_to_delete), collection
                )
                vs.delete(ids_to_delete)
                # Save after delete
                if self._persistent:
                    vs.save_local(str(self._index_root / collection))

        # Handle Upserts (Add new)
        documents_to_add = []
        for loaded_doc in docs_to_upsert:
            file_name = os.path.basename(loaded_doc.path)
            metadata = {
                "source": file_name,
                "last_modification": loaded_doc.last_modified
            }
            
            try:
                if (location_config and location_config.metadata and location_config.metadata.structure):
                    file_name_no_ext = Path(file_name).stem
                    parts = file_name_no_ext.split(location_config.metadata.delimiter)
                    for i, prop_setting in enumerate(location_config.metadata.structure):
                        if i < len(parts):
                            metadata[prop_setting.name] = parts[i]
            except Exception as e:
                logger.warning("Metadata extraction failed for %s: %s", file_name, e)

            documents_to_add.append(Document(page_content=loaded_doc.content, metadata=metadata))

        if documents_to_add:
            chunk_size = 500
            chunk_overlap = 50
            if location_config and location_config.loader_config:
                chunk_size = location_config.loader_config.get("chunk_size", 500)
                chunk_overlap = location_config.loader_config.get("chunk_overlap", 50)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            docs = text_splitter.split_documents(documents_to_add)
            
            logger.info(
                "Adding %d chunks to FAISS collection '%s'", len(docs), collection
            )
            
            if vs is None:
                # Create new index
                self._vectorstores[collection] = FAISS.from_documents(docs, self._embeddings)
            else:
                vs.add_documents(docs)
            
            # Save to disk
            if self._persistent:
                self._vectorstores[collection].save_local(str(self._index_root / collection))

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        for loader_info in self._loaders:
            if loader_info['poll_interval'] == 0:
                logger.info(
                    "Skipping periodic synchronization watcher, collection: %s",
                    loader_info['collection']
                )
                continue
            update_callback = partial(self._update_vector_store, collection=loader_info['collection'])
            loader_info["loader"].watcher(callback=update_callback)
            logger.info("Started watcher for %s", loader_info['collection'])

    def stop_periodic_sync(self):
        """Stops the background synchronization thread gracefully."""
        for loader_info in self._loaders:
            loader_info["loader"].stop_watcher()
            loader_info["loader"].close()

    def get_vectorstore(self, collection: str) -> Optional[FAISS]:
        """Returns the FAISS vector store instance for a given collection."""
        return self._vectorstores.get(collection)

    def name(self) -> str:
        """Returns the name of the vector store manager."""
        return self._name

    def close(self):
        """Closes client connections and stops background threads."""
        self.stop_periodic_sync()
        logger.info("Faiss manager closed.")
