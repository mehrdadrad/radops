"""
This module implements the ChromaVectorStoreManager for managing ChromaDB vector stores.
"""
import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from integrations.github.github_loader import GithubLoader
from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class ChromaVectorStoreManager:
    """
    Manages a ChromaDB vector store, ensuring it's synchronized with a directory of text files.
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
        Initializes the manager, connects to ChromaDB, and synchronizes the vector store.

        Args:
            name: The name of the vector store manager instance.
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
            skip_initial_sync: If True, skips the initial data synchronization.
        """
        self._config = settings.vector_store.providers["chroma"]
        try:
            self._client = chromadb.PersistentClient(
                path=self._config.get("path", ".chromadb")
            )
            logger.info("Successfully connected to ChromaDB.")
            logger.info("  - ChromaDB Version: %s", chromadb.__version__)
        except Exception as e:
            logger.error("Failed to connect to ChromaDB: %s", e, exc_info=True)
            raise

        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._name = name
        self._stop_event = threading.Event()
        self._vectorstores = {}
        self._loaders = []

        for location in self._sync_locations:
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
                        "collection": location.collection,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[location.collection] = Chroma(
                    client=self._client,
                    collection_name=location.collection,
                    embedding_function=embeddings
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path],
                    poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": location.collection,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[location.collection] = Chroma(
                    client=self._client,
                    collection_name=location.collection,
                    embedding_function=embeddings
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
                        "collection": location.collection,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[location.collection] = Chroma(
                    client=self._client,
                    collection_name=location.collection,
                    embedding_function=embeddings
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
            logger.info("Loading initial data for %s...", collection_name)
            self._update_vector_store(loader.load_data(), collection_name)

    def _update_vector_store(
        self, changed_docs: List[LoadedDocument], collection: str
    ):
        """
        Updates the ChromaDB collection with changed documents.
        """
        vectorstore = self._vectorstores.get(collection)
        if not vectorstore:
            logger.error("No vector store found for collection '%s'.", collection)
            return

        location_config = next(
            (loc for loc in self._sync_locations
             if loc.collection == collection),
            None
        )

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
                "Removing %d deleted files from vector store: %s",
                len(deleted_sources),
                ', '.join(sorted(deleted_sources))
            )
            vectorstore.delete(where={"source": {"$in": deleted_sources}})

        # 2. Handle upserts: check last_modified to avoid redundant updates
        docs_that_need_update = []
        if docs_to_upsert:
            upsert_sources = [
                os.path.basename(doc.path) for doc in docs_to_upsert
            ]

            # Fetch existing docs to compare last_modified timestamps
            existing_docs = vectorstore.get(
                where={"source": {"$in": upsert_sources}},
                include=["metadatas"]
            )
            existing_docs_map = {
                meta['source']: meta.get('last_modification')
                for meta in existing_docs.get('metadatas', []) if meta
            }

            for doc in docs_to_upsert:
                source_name = os.path.basename(doc.path)
                if (source_name not in existing_docs_map or
                        doc.last_modified != existing_docs_map.get(source_name)):
                    docs_that_need_update.append(doc)
                else:
                    logger.debug(
                        "Skipping update for '%s', last_modified timestamp is unchanged.",
                        source_name
                    )

            if docs_that_need_update:
                sources_to_update = [
                    os.path.basename(doc.path) for doc in docs_that_need_update
                ]
                logger.info(
                    "Upserting %d files in vector store",
                    len(sources_to_update)
                )
                # In Chroma, `add_documents` acts as an upsert if IDs are same.
                # To be safe and mimic Weaviate's delete-then-add for updates,
                # we delete first.
                vectorstore.delete(
                    where={"source": {"$in": sources_to_update}}
                )
                docs_to_upsert = docs_that_need_update
            else:
                docs_to_upsert = []

        # 3. Prepare and add new/updated documents
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
                    "Could not extract metadata from filename '%s': %s",
                    file_name, e
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
                "Adding %d document chunks to the vector store...", len(docs)
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

    def get_vectorstore(self, collection: str) -> Chroma:
        """Returns the Chroma vector store instance for a given collection."""
        return self._vectorstores.get(collection)

    def name(self) -> str:
        """Returns the name of the vector store manager."""
        return self._name

    def close(self):
        """Closes client connections and stops background threads."""
        self.stop_periodic_sync()
        # ChromaDB's PersistentClient doesn't have an explicit close method.
        # The connection is managed by the client library.
        logger.info("ChromaDB manager closed.")