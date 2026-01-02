"""
This module implements the MilvusVectorStoreManager for managing Milvus vector stores.
"""
import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import List

# Suppress gRPC warnings associated with Milvus/pymilvus
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

import pymilvus
from langchain_milvus import Milvus
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from integrations.github.github_loader import GithubLoader
from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class MilvusVectorStoreManager:
    """
    Manages a Milvus vector store, ensuring it's synchronized with a directory of text files.
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
        Initializes the manager, connects to Milvus, and synchronizes the vector store.

        Args:
            name: The name of the vector store manager instance.
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
            skip_initial_sync: If True, skips the initial data synchronization.
        """
        self._config = settings.vector_store.providers["milvus"]
        self._uri = self._config.get("uri")
        self._token = self._config.get("token")
        self._user = self._config.get("user")
        self._password = self._config.get("password")

        self._connection_args = {
            "uri": self._uri,
            "token": self._token,
            "user": self._user,
            "password": self._password,
        }
        # Clean up None values
        self._connection_args = {k: v for k, v in self._connection_args.items() if v is not None}

        try:
            self._client = MilvusClient(**self._connection_args)
            logger.info("Successfully connected to Milvus.")
            logger.info(
                f"  - Milvus Server Version: {self._client.get_server_version()}"
            )
            logger.info(f"  - Milvus Client Version: {pymilvus.__version__}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
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
                        "collection": collection_name,
                        "poll_interval": poll_interval
                    }
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path],
                    poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": collection_name,
                        "poll_interval": poll_interval
                    }
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
                        "collection": collection_name,
                        "poll_interval": poll_interval
                    }
                )
            else:
                logger.warning(f"Unsupported sync location type: {location.type}")
                continue

            # Initialize LangChain vectorstore
            self._vectorstores[collection_name] = Milvus(
                embedding_function=embeddings,
                connection_args=self._connection_args,
                collection_name=collection_name,
                auto_id=True,
            )

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
            logger.info(f"Loading initial data for collection '{collection_name}'...")
            self._update_vector_store(loader.load_data(), collection_name)

    def _update_vector_store(
        self, changed_docs: List[LoadedDocument], collection: str
    ):
        """
        Updates the Milvus collection with changed documents.
        """
        vectorstore = self._vectorstores.get(collection)
        if not vectorstore:
            logger.error(f"No vector store found for collection '{collection}'.")
            return

        location_config = next(
            (loc for loc in self._sync_locations
             if loc.collection == collection),
            None
        )

        # Check if collection exists to avoid errors on first run
        collection_exists = self._client.has_collection(collection)

        docs_to_upsert = [
            doc for doc in changed_docs if doc.content is not None
        ]
        docs_to_delete = [doc for doc in changed_docs if doc.content is None]

        # Filter out documents that are already up-to-date in Milvus
        if docs_to_upsert and collection_exists:
            docs_that_need_update = []
            for doc in docs_to_upsert:
                source = os.path.basename(doc.path)
                try:
                    # Milvus scalar query
                    filter_expr = f'source == "{source}"'
                    results = self._client.query(
                        collection_name=collection,
                        filter=filter_expr,
                        output_fields=["last_modification"],
                        limit=1
                    )
                    
                    needs_update = True
                    if results:
                        stored_mod = results[0].get("last_modification")
                        if stored_mod is not None:
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
                        f"Error checking existence of '{source}' in Milvus: {e}"
                    )
                    docs_that_need_update.append(doc)
            docs_to_upsert = docs_that_need_update

        # Handle deletions
        if docs_to_delete and collection_exists:
            deleted_sources = [
                os.path.basename(doc.path) for doc in docs_to_delete
            ]
            logger.info(
                f"Removing {len(deleted_sources)} deleted files from collection '{collection}': "
                f"{', '.join(sorted(deleted_sources))}"
            )
            try:
                expr = f'source in {str(deleted_sources)}'
                self._client.delete(collection_name=collection, filter=expr)
            except Exception as e:
                logger.warning(
                    f"Error deleting from Milvus collection "
                    f"'{collection}': {e}"
                )

        # Handle upserts: Delete existing chunks for these files first to avoid duplicates
        if docs_to_upsert and collection_exists:
            upsert_sources = [
                os.path.basename(doc.path) for doc in docs_to_upsert
            ]
            logger.info(
                f"Upserting {len(upsert_sources)} files in collection '{collection}': "
                f"{', '.join(sorted(upsert_sources))}"
            )
            try:
                expr = f'source in {str(upsert_sources)}'
                self._client.delete(collection_name=collection, filter=expr)
            except Exception as e:
                logger.warning(
                    f"Error deleting existing vectors in Milvus "
                    f"collection '{collection}': {e}"
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
                    f"Could not extract metadata from filename "
                    f"'{file_name}': {e}"
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
                f"Adding {len(docs)} document chunks to collection '{collection}'..."
            )
            vectorstore.add_documents(docs)

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        for loader_info in self._loaders:
            if loader_info['poll_interval'] == 0:
                logger.info(
                    f"Skipping periodic synchronization watcher, collection: {loader_info['collection']}"
                )
                continue
            update_callback = partial(
                self._update_vector_store,
                collection=loader_info['collection']
            )
            loader_info["loader"].watcher(callback=update_callback)
            logger.info(
                f"Starting periodic synchronization watcher, collection: {loader_info['collection']}"
            )

    def stop_periodic_sync(self):
        """Stops the background synchronization thread gracefully."""
        logger.info("Stopping periodic synchronization watcher...")
        for loader_info in self._loaders:
            loader_info["loader"].stop_watcher()
            loader_info["loader"].close()

    def get_vectorstore(self, collection: str) -> Milvus:
        """Returns the Milvus vector store instance for a given collection."""
        return self._vectorstores.get(collection)

    def name(self) -> str:
        return self._name

    def close(self):
        """Closes client connections and stops background threads."""
        self.stop_periodic_sync()
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Milvus manager closed.")
