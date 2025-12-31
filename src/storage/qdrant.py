"""
This module implements the QdrantVectorStoreManager for managing Qdrant vector stores.
"""
import logging
import os
import asyncio
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import List

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from integrations.github.github_loader import GithubLoader
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
        sync_interval: int = 60
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
        self._name = name
        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._vectorstores = {}
        self._loaders = []

        self._client = self._connect_to_qdrant()
        self._initialize_collections()

        self._initial_sync()

        if self._sync_interval > 0:
            self.start_periodic_sync()

    def _get_connection_params(self):
        """Extracts connection parameters from settings."""
        # pylint: disable=no-member
        config = settings.vector_store.providers.get("qdrant", {})
        return {
            "url": config.get("url"),
            "host": config.get("host"),
            "port": config.get("port"),
            "path": config.get("path"),
            "api_key": config.get("api_key"),
            "prefer_grpc": config.get("prefer_grpc", False),
        }

    def _connect_to_qdrant(self) -> QdrantClient:
        """Connects to the Qdrant client using settings."""
        params = self._get_connection_params()
        try:
            client = QdrantClient(**params)
            logger.info("Successfully connected to Qdrant.")
            return client
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to connect to Qdrant: %s", e)
            raise

    def _connect_to_qdrant_async(self) -> AsyncQdrantClient:
        """Connects to the Qdrant async client using settings."""
        params = self._get_connection_params()
        try:
            client = AsyncQdrantClient(**params)
            return client
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to connect to Qdrant Async: %s", e)
            raise

    def _initialize_collections(self):
        """Initializes loaders and collections for sync locations."""
        for location in self._sync_locations:
            collection_name = location.collection

            loader = self._create_loader(location)
            if not loader:
                continue

            self._loaders.append(
                {"loader": loader, "collection": collection_name}
            )

            self._ensure_collection_exists(collection_name)
            self._ensure_payload_indices(collection_name, location)

            # QdrantVectorStore will create the collection if it doesn't exist
            # when we add documents, but we initialize it here.
            self._vectorstores[collection_name] = QdrantVectorStore(
                client=self._client,
                collection_name=collection_name,
                embedding=self._embeddings
            )

    def _create_loader(self, location: SyncLocationSettings):
        """Creates a loader based on location type."""
        if location.type == "fs":
            return FileSystemLoader(
                path=location.path,
                poll_interval=location.sync_interval
            )
        if location.type == "gdrive":
            return GoogleDriveLoader(
                folder_ids=[location.path],
                poll_interval=location.sync_interval
            )
        if location.type == "github":
            return GithubLoader(
                repo_names=location.path.split(","),
                loader_config=location.loader_config,
                poll_interval=location.sync_interval
            )

        logger.warning("Unsupported sync location type: %s", location.type)
        return None

    def _ensure_collection_exists(self, collection_name: str):
        """Creates the collection if it does not exist."""
        if not self._client.collection_exists(collection_name):
            logger.info("Collection '%s' does not exist. Creating it...", collection_name)
            try:
                # Infer dimension from embedding model
                sample_embedding = self._embeddings.embed_query("test")
                vector_size = len(sample_embedding)
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Failed to create collection '%s': %s", collection_name, e
                )

    def _ensure_payload_indices(self, collection_name: str, location: SyncLocationSettings):
        """Ensures payload indices exist for the collection."""
        fields_to_index = ["source", "last_modification"]
        if location.metadata and location.metadata.structure:
            fields_to_index.extend(
                [prop.name for prop in location.metadata.structure]
            )

        for field in fields_to_index:
            field_schema = models.PayloadSchemaType.KEYWORD
            if field == "last_modification":
                field_schema = models.PayloadSchemaType.INTEGER

            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=f"metadata.{field}",
                    field_schema=field_schema,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "Payload index creation for '%s' metadata.%s: %s",
                    collection_name, field, e
                )

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
            logger.info(
                "Loading initial data for collection '%s'...", collection_name
            )
            self._update_vector_store(loader.load_data(), collection_name)

    async def _check_doc_exists(
        self, client: AsyncQdrantClient, doc: LoadedDocument, collection: str
    ) -> bool:
        """Checks if a document exists and is unchanged in Qdrant."""
        source = os.path.basename(doc.path)
        try:
            count_result = await client.count(
                collection_name=collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source)
                        ),
                        models.FieldCondition(
                            key="metadata.last_modification",
                            match=models.MatchValue(value=doc.last_modified)
                        )
                    ]
                ),
                exact=True
            )
            if count_result.count > 0:
                logger.debug(
                    "Skipping update for '%s', last_modification timestamp is unchanged.",
                    source
                )
                return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Error checking existence of '%s' in Qdrant: %s", source, e
            )
        return False

    async def _filter_unchanged_docs_async(
        self, docs: List[LoadedDocument], collection: str
    ) -> List[LoadedDocument]:
        """Async version of filtering unchanged docs."""
        client = self._connect_to_qdrant_async()
        try:
            tasks = [
                self._check_doc_exists(client, doc, collection) for doc in docs
            ]
            results = await asyncio.gather(*tasks)
            return [doc for doc, exists in zip(docs, results) if not exists]
        finally:
            await client.close()

    def _filter_unchanged_docs(
        self, docs: List[LoadedDocument], collection: str
    ) -> List[LoadedDocument]:
        """Filters out documents that haven't changed in Qdrant."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Run in a separate thread to avoid blocking the event loop
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self._filter_unchanged_docs_async(docs, collection)
                ).result()

        return asyncio.run(self._filter_unchanged_docs_async(docs, collection))

    def _delete_docs(self, docs: List[LoadedDocument], collection: str):
        """Deletes documents from Qdrant."""
        deleted_sources = [
            os.path.basename(doc.path) for doc in docs
        ]
        logger.info(
            "Removing %d deleted files from collection '%s': %s",
            len(deleted_sources), collection, ', '.join(sorted(deleted_sources))
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

    def _upsert_docs(
        self,
        docs: List[LoadedDocument],
        collection: str,
        vectorstore: QdrantVectorStore
    ):
        """Upserts documents into Qdrant."""
        upsert_sources = [
            os.path.basename(doc.path) for doc in docs
        ]
        logger.info(
            "Upserting %d files in collection '%s': %s",
            len(upsert_sources), collection, ', '.join(sorted(upsert_sources))
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

        # Prepare and add new/updated documents
        documents_to_add = []
        for loaded_doc in docs:
            doc = self._prepare_document(loaded_doc, collection)
            documents_to_add.append(doc)

        if documents_to_add:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            docs_chunks: List[Document] = text_splitter.split_documents(
                documents_to_add
            )
            logger.info(
                "Adding %d document chunks to collection '%s'...",
                len(docs_chunks), collection
            )
            vectorstore.add_documents(docs_chunks)

    def _prepare_document(self, loaded_doc: LoadedDocument, collection: str) -> Document:
        """Prepares a Document object with metadata."""
        file_name = os.path.basename(loaded_doc.path)
        metadata = {
            "source": file_name,
            "last_modification": loaded_doc.last_modified
        }

        location_config = next(
            (loc for loc in self._sync_locations
             if loc.collection == collection),
            None
        )

        try:
            if (location_config and
                    location_config.metadata and
                    location_config.metadata.structure):
                file_name_no_ext = Path(file_name).stem
                parts = file_name_no_ext.split(
                    location_config.metadata.delimiter
                )
                for i, prop_setting in enumerate(location_config.metadata.structure):
                    if i < len(parts):
                        metadata[prop_setting.name] = parts[i]
        except (ValueError, AttributeError) as e:
            logger.warning(
                "Could not extract metadata from filename '%s': %s", file_name, e
            )

        return Document(page_content=loaded_doc.content, metadata=metadata)

    def _update_vector_store(
        self, changed_docs: List[LoadedDocument], collection: str
    ):
        """
        Updates the Qdrant collection with changed documents.
        """
        vectorstore = self._vectorstores.get(collection)
        if not vectorstore:
            logger.error("No vector store found for collection '%s'.", collection)
            return

        docs_to_upsert = [
            doc for doc in changed_docs if doc.content is not None
        ]
        docs_to_delete = [doc for doc in changed_docs if doc.content is None]

        # Filter out documents that are already up-to-date in Qdrant
        if docs_to_upsert:
            docs_to_upsert = self._filter_unchanged_docs(docs_to_upsert, collection)

        # 1. Handle deletions
        if docs_to_delete:
            self._delete_docs(docs_to_delete, collection)

        # 2. Handle upserts
        if docs_to_upsert:
            self._upsert_docs(docs_to_upsert, collection, vectorstore)

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        logger.info(
            "Starting periodic synchronization watcher every %s seconds.",
            self._sync_interval
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
        """Returns the name of the vector store manager."""
        return self._name

    def close(self):
        """Closes client connections and stops background threads."""
        self.stop_periodic_sync()
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Qdrant manager closed.")
        