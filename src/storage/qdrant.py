import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import List

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class QdrantVectorStoreManager:
    """
    Manages a Qdrant vector store, ensuring it's synchronized with a directory of text files.
    """

    def __init__(
        self,
        name: str,
        sync_locations: list[SyncLocationSettings],
        embeddings,
        sync_interval: int = 10
    ):
        """
        Initializes the manager, connects to Qdrant, and synchronizes the vector store.

        Args:
            name: The name of the vector store manager instance.
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
        """
        self._config = settings.vector_store.providers.get("qdrant", {})
        
        # Extract connection params
        self._url = self._config.get("url")
        self._host = self._config.get("host")
        self._port = self._config.get("port")
        self._path = self._config.get("path")
        self._api_key = self._config.get("api_key")
        self._prefer_grpc = self._config.get("prefer_grpc", False)

        try:
            self._client = QdrantClient(
                url=self._url,
                host=self._host,
                port=self._port,
                path=self._path,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc
            )
            logger.info("Successfully connected to Qdrant.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
            raise

        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._name = name
        self._stop_event = threading.Event()
        self._vectorstores = {}
        self._loaders = []

        for location in self._sync_locations:
            collection_name = location.collection
            
            if location.type == "fs":
                loader = FileSystemLoader(
                    path=location.path,
                    poll_interval=location.sync_interval
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path],
                    poll_interval=location.sync_interval
                )
            else:
                logger.warning(f"Unsupported sync location type: {location.type}")
                continue

            self._loaders.append(
                {"loader": loader, "collection": collection_name}
            )
            
            # QdrantVectorStore will create the collection if it doesn't exist
            # when we add documents, but we initialize it here.
            self._vectorstores[collection_name] = QdrantVectorStore(
                client=self._client,
                collection_name=collection_name,
                embedding=embeddings
            )

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
            logger.info(f"Loading initial data for namespace '{collection_name}'...")
            self._update_vector_store(loader.load_data(), collection_name)

    def _update_vector_store(
        self, changed_docs: List[LoadedDocument], collection: str
    ):
        """
        Updates the Qdrant collection with changed documents.
        """
        vectorstore = self._vectorstores.get(collection)
        if not vectorstore:
            logger.error(f"No vector store found for collection '{collection}'.")
            return

        docs_to_upsert = [
            doc for doc in changed_docs if doc.content is not None
        ]
        docs_to_delete = [doc for doc in changed_docs if doc.content is None]

        # 1. Handle deletions
        if docs_to_delete:
            deleted_sources = [
                os.path.basename(doc.path) for doc in docs_to_delete
            ]
            logger.info(
                f"Removing {len(deleted_sources)} deleted files from collection '{collection}': "
                f"{', '.join(sorted(deleted_sources))}"
            )
            # Use client directly for robust filtering
            self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchAny(any=deleted_sources)
                            )
                        ]
                    )
                )
            )

        # 2. Handle upserts: Delete existing chunks for these files first to avoid duplicates
        if docs_to_upsert:
            upsert_sources = [
                os.path.basename(doc.path) for doc in docs_to_upsert
            ]
            logger.info(
                f"Upserting {len(upsert_sources)} files in collection '{collection}': "
                f"{', '.join(sorted(upsert_sources))}"
            )
            # Delete existing vectors for these sources before re-adding
            self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchAny(any=upsert_sources)
                            )
                        ]
                    )
                )
            )

        # 3. Prepare and add new/updated documents
        documents_to_add = []
        for loaded_doc in docs_to_upsert:
            file_name = os.path.basename(loaded_doc.path)
            metadata = {
                "source": file_name,
                "last_modification": loaded_doc.last_modified
            }
            # Metadata extraction logic (same as other providers)
            # ... (omitted for brevity, assume standard logic if needed, or copy from Pinecone)
            
            doc = Document(page_content=loaded_doc.content, metadata=metadata)
            documents_to_add.append(doc)

        if documents_to_add:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            docs: List[Document] = text_splitter.split_documents(
                documents_to_add
            )
            logger.info(
                f"Adding {len(docs)} document chunks to collection '{collection}'..."
            )
            vectorstore.add_documents(docs)

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        logger.info(
            f"Starting periodic synchronization watcher every "
            f"{self._sync_interval} seconds."
        )
        for loader_info in self._loaders:
            update_callback = partial(
                self._update_vector_store,
                collection=loader_info['collection']
            )
            loader_info["loader"].watcher(callback=update_callback)

    def stop_periodic_sync(self):
        """Stops the background synchronization thread gracefully."""
        logger.info("Stopping periodic synchronization watcher...")
        for loader_info in self._loaders:
            loader_info["loader"].stop_watcher()
            loader_info["loader"].close()

    def get_vectorstore(self, collection: str) -> QdrantVectorStore:
        """Returns the Qdrant vector store instance for a given collection."""
        return self._vectorstores.get(collection)

    def name(self) -> str:
        return self._name

    def close(self):
        """Closes client connections and stops background threads."""
        self.stop_periodic_sync()
        if self._client:
            self._client.close()
        logger.info("Qdrant manager closed.")