import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import List

import weaviate
import weaviate.classes as wvc
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore as Weaviate

from config.config import SyncLocationSettings, settings
from integrations.fs.fs_loader import FileSystemLoader
from integrations.google.gdrive_loader import GoogleDriveLoader
from integrations.github.github_loader import GithubLoader
from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class WeaviateVectorStoreManager:
    """
    Manages a Weaviate vector store, ensuring it's synchronized with a directory of text files.
    """
    def __init__(
        self,
        name: str,
        sync_locations: list[SyncLocationSettings],
        embeddings,
        sync_interval: int = 10
    ):
        """
        Initializes the manager, connects to Weaviate, and synchronizes the vector store.

        Args:
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
        """

        self._config = settings.vector_store.providers["weaviate"]
        try:
            self._client = weaviate.connect_to_custom(
                http_host=self._config.get("http_host", "localhost"),
                http_port=self._config.get("http_port", 8080),
                http_secure=self._config.get("http_secure", False),
                grpc_host=self._config.get("grpc_host", "localhost"),
                grpc_port=self._config.get("grpc_port", 50051),
                grpc_secure=self._config.get("grpc_secure", False),
            )
            if self._client.is_ready():
                meta = self._client.get_meta()
                server_version = meta.get("version", "unknown")
                logger.info("Successfully connected to Weaviate.")
                logger.info(f"  - Weaviate Server Version: {server_version}")
                logger.info(f"  - Weaviate Client Version: {weaviate.__version__}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}", exc_info=True)
            raise

        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._name = name
        self._vectorstores = {}
        self._loaders = []

        for location in self._sync_locations:
            if location.type == "fs":
                loader = FileSystemLoader(
                    path=location.path, poll_interval=location.sync_interval
                )
                self._loaders.append(
                    {"loader": loader, "collection": location.collection}
                )
                self._vectorstores[location.collection] = Weaviate(
                    self._client, location.collection, "text",
                    embedding=embeddings,
                    attributes=["source", "last_modification"]
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path], poll_interval=location.sync_interval
                )
                self._loaders.append(
                    {"loader": loader, "collection": location.collection}
                )
                self._vectorstores[location.collection] = Weaviate(
                    self._client, location.collection, "text",
                    embedding=embeddings,
                    attributes=["source", "last_modification"]
                )
            elif location.type == "github":
                loader = GithubLoader(
                    repo_names=location.path.split(","),
                    loader_config=location.loader_config,
                    poll_interval=location.sync_interval
                )
                self._loaders.append(
                    {"loader": loader, "collection": location.collection}
                )
                self._vectorstores[location.collection] = Weaviate(
                    self._client, location.collection, "text",
                    embedding=embeddings,
                    attributes=["source", "last_modification"]
                )
            else:
                logger.warning(f"Unsupported sync location type: {location.type}")

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
        Performs a full synchronization by clearing and re-populating the collection.
        """
        logger.info("Performing initial data synchronization for all locations...")
        for loader in self._loaders:
            logger.info(f"Loading initial data for {loader['collection']}...")
            self._update_vector_store(
                loader["loader"].load_data(),
                loader['collection'],
                full_refresh=False
            )

    def _update_vector_store(
        self,
        changed_docs: List[LoadedDocument],
        collection: str,
        full_refresh: bool = False
    ):
        """
        Updates the Weaviate collection. Can perform a full refresh or a granular update.
        """
        docs_to_upsert = []
        if full_refresh:
            # Clear existing collection for a full refresh
            if self._client.collections.exists(collection):
                logger.info(f"Clearing Weaviate collection: {collection}")
                self._client.collections.delete(collection)

            # Re-create collection
            logger.info(f"Creating new Weaviate collection: {collection}")

            location_config = next(
                (loc for loc in self._sync_locations
                 if loc.collection == collection),
                None
            )
            custom_metadata_properties = []
            if (location_config and
                    location_config.metadata and
                    location_config.metadata.structure):
                for prop_setting in location_config.metadata.structure:
                    custom_metadata_properties.append(
                        wvc.config.Property(
                            name=prop_setting.name,
                            description=prop_setting.description,
                            data_type=wvc.config.DataType.TEXT)
                    )
            else:
                logger.warning(
                    f"No metadata structure found for collection '{collection}' "
                    "in config.yaml. No custom properties will be created."
                )

            try:
                self._client.collections.create(
                    name=collection,
                    properties=[
                        wvc.config.Property(
                            name="text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(
                            name="source", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(
                            name="last_modification", data_type=wvc.config.DataType.NUMBER),
                    ] + custom_metadata_properties,
                )
            except Exception as e:
                logger.error(f"Failed to create collection: {e}", exc_info=True)
                return
            # For a full refresh, all changed docs are candidates for upserting
            docs_to_upsert = [doc for doc in changed_docs if doc.content is not None]
        else:
            # Granular update: separate deleted from modified/added files
            docs_to_upsert = [doc for doc in changed_docs if doc.content is not None]
            docs_to_delete = [doc for doc in changed_docs if doc.content is None]

            if not self._client.collections.exists(collection):
                logger.warning(
                    f"Collection '{collection}' not found for granular update. "
                    "Performing full refresh logic instead."
                )
                self._update_vector_store(changed_docs, collection, full_refresh=True)
                return

            collection_obj = self._client.collections.get(collection)

            # Check for schema compatibility
            try:
                if "source" not in {p.name for p in collection_obj.config.get().properties}:
                    logger.warning(
                        f"Collection '{collection}' is missing 'source' property. "
                        "Performing full refresh to fix schema."
                    )
                    self._update_vector_store(changed_docs, collection, full_refresh=True)
                    return
            except Exception as e:
                logger.warning(f"Failed to validate schema for '{collection}': {e}")

            # 1. Handle deletions
            if docs_to_delete:
                deleted_sources = [os.path.basename(doc.path) for doc in docs_to_delete]
                logger.info(
                    f"Removing {len(deleted_sources)} deleted files from vector store: "
                    f"{', '.join(sorted(deleted_sources))}"
                )
                collection_obj.data.delete_many(
                    where=wvc.query.Filter.by_property("source").contains_any(deleted_sources)
                )

            # 2. Handle upserts: check last_modified to avoid redundant updates
            if docs_to_upsert:
                upsert_sources = [os.path.basename(doc.path) for doc in docs_to_upsert]

                # Fetch existing docs to compare last_modified timestamps
                response = collection_obj.query.fetch_objects(
                    filters=wvc.query.Filter.by_property("source").contains_any(upsert_sources),
                    return_properties=["source", "last_modification"]
                )
                existing_docs_map = {
                    obj.properties['source']: obj.properties['last_modification']
                    for obj in response.objects
                }

                docs_that_need_update = []
                for doc in docs_to_upsert:
                    source_name = os.path.basename(doc.path)
                    if (source_name not in existing_docs_map or
                            doc.last_modified != existing_docs_map.get(source_name, 0)):
                        docs_that_need_update.append(doc)
                    else:
                        logger.debug(
                            f"Skipping update for '{source_name}', "
                            "last_modified timestamp is unchanged."
                        )

                if docs_that_need_update:
                    sources_to_update = [
                        os.path.basename(doc.path) for doc in docs_that_need_update
                    ]
                    logger.info(
                        f"Upserting {len(sources_to_update)} files in vector store: "
                        f"{', '.join(sorted(sources_to_update))}"
                    )
                    collection_obj.data.delete_many(
                        where=wvc.query.Filter.by_property("source").contains_any(sources_to_update)
                    )
                    docs_to_upsert = docs_that_need_update  # Only process documents that need updating
                else:
                    docs_to_upsert = []  # No documents to update

        # 3. Prepare and add new documents
        documents_to_add = []
        for loaded_doc in docs_to_upsert:
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
                    f"Could not extract metadata from filename '{file_name}': {e}"
                )

            doc = Document(page_content=loaded_doc.content, metadata=metadata)
            documents_to_add.append(doc)

        if documents_to_add:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            docs: List[Document] = text_splitter.split_documents(
                documents_to_add
            )
            logger.info(f"Adding {len(docs)} document chunks to the vector store...")
            self._vectorstores[collection].add_documents(docs)

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        logger.info(f"Starting periodic synchronization watcher every {self._sync_interval} seconds.")
        for loader in self._loaders:
            # Use functools.partial to create a callback with the collection
            # name pre-filled. The loader's watcher expects a callback that
            # only takes one argument (the documents).
            update_callback = partial(
                self._update_vector_store,
                collection=loader['collection']
            )
            loader["loader"].watcher(callback=update_callback)

    def stop_periodic_sync(self):
        """Stops the background synchronization thread gracefully."""
        logger.info("Stopping periodic synchronization watcher...")
        for loader in self._loaders:
            loader["loader"].stop_watcher()
            loader["loader"].close()

    def get_vectorstore(self, collection: str) -> Weaviate:
        """
        Returns the Weaviate vector store instance.

        Returns:
            The Weaviate vector store instance.
        """
        return self._vectorstores[collection]

    def name(self) -> str:
        return self._name

    def close(self):
        """
        Closes the Weaviate client connection.
        """
        self.stop_periodic_sync()
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Weaviate client connection closed.")
