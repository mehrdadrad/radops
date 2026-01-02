"""
This module implements the WeaviateVectorStoreManager for managing Weaviate vector stores.
"""
import logging
import os
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
        sync_interval: int = 60,
        skip_initial_sync: bool = False
    ):
        """
        Initializes the manager, connects to Weaviate, and synchronizes the vector store.

        Args:
            sync_locations: A list of locations to sync from.
            embeddings: The embeddings model to use for vectorization.
            sync_interval: The interval in seconds for the loader to check for file changes.
                           Set to 0 to disable periodic syncing.
            skip_initial_sync: If True, skips the initial data synchronization.
        """

        config = settings.vector_store.providers["weaviate"]
        try:
            self._client = weaviate.connect_to_custom(
                http_host=config.get("http_host", "localhost"),
                http_port=config.get("http_port", 8080),
                http_secure=config.get("http_secure", False),
                grpc_host=config.get("grpc_host", "localhost"),
                grpc_port=config.get("grpc_port", 50051),
                grpc_secure=config.get("grpc_secure", False),
            )
            if self._client.is_ready():
                meta = self._client.get_meta()
                server_version = meta.get("version", "unknown")
                logger.info("Successfully connected to Weaviate.")
                logger.info("  - Weaviate Server Version: %s", server_version)
                logger.info("  - Weaviate Client Version: %s", weaviate.__version__)
        except Exception as e:
            logger.error("Failed to connect to Weaviate: %s", e, exc_info=True)
            raise

        self._sync_locations = sync_locations
        self._embeddings = embeddings
        self._sync_interval = sync_interval
        self._name = name
        self._vectorstores = {}
        self._loaders = []

        for location in self._sync_locations:
            # Ensure the collection exists with the correct schema
            if not self._client.collections.exists(location.collection):
                self._create_collection(location.collection)

            poll_interval = (
                location.sync_interval
                if location.sync_interval is not None
                else self._sync_interval
            )
            if location.type == "fs":
                loader = FileSystemLoader(
                    path=location.path, poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": location.collection,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[location.collection] = Weaviate(
                    self._client, location.collection, "text",
                    embedding=embeddings,
                    attributes=["source", "last_modification"]
                )
            elif location.type == "gdrive":
                loader = GoogleDriveLoader(
                    folder_ids=[location.path], poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": location.collection,
                        "poll_interval": poll_interval
                    }
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
                    poll_interval=poll_interval
                )
                self._loaders.append(
                    {
                        "loader": loader,
                        "collection": location.collection,
                        "poll_interval": poll_interval
                    }
                )
                self._vectorstores[location.collection] = Weaviate(
                    self._client, location.collection, "text",
                    embedding=embeddings,
                    attributes=["source", "last_modification"]
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
        Performs a full synchronization by clearing and re-populating the collection.
        """
        logger.info("Performing initial data synchronization for all locations...")
        for loader in self._loaders:
            logger.info("Loading initial data for %s...", loader['collection'])
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
        Updates the Weaviate collection.
        """
        if full_refresh:
            self._perform_full_refresh(collection, changed_docs)
        else:
            self._perform_granular_update(collection, changed_docs)

    def _perform_full_refresh(self, collection: str, changed_docs: List[LoadedDocument]):
        """Performs a full refresh of the collection."""
        if self._client.collections.exists(collection):
            logger.info("Clearing Weaviate collection: %s", collection)
            self._client.collections.delete(collection)

        self._create_collection(collection)

        docs_to_upsert = [doc for doc in changed_docs if doc.content is not None]
        if docs_to_upsert:
            self._add_documents_to_collection(collection, docs_to_upsert)

    def _perform_granular_update(self, collection: str, changed_docs: List[LoadedDocument]):
        """Performs a granular update of the collection."""
        docs_to_upsert = [doc for doc in changed_docs if doc.content is not None]
        docs_to_delete = [doc for doc in changed_docs if doc.content is None]

        if not self._client.collections.exists(collection):
            logger.warning(
                "Collection '%s' not found for granular update. "
                "Performing full refresh logic instead.",
                collection
            )
            self._perform_full_refresh(collection, changed_docs)
            return

        collection_obj = self._client.collections.get(collection)

        # Check for schema compatibility
        try:
            if "source" not in {p.name for p in collection_obj.config.get().properties}:
                logger.warning(
                    "Collection '%s' is missing 'source' property. "
                    "Performing full refresh to fix schema.",
                    collection
                )
                self._perform_full_refresh(collection, changed_docs)
                return
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to validate schema for '%s': %s", collection, e)

        if docs_to_delete:
            self._delete_documents(collection_obj, docs_to_delete)

        if docs_to_upsert:
            docs_to_add = self._filter_upserts(collection_obj, docs_to_upsert)
            if docs_to_add:
                self._add_documents_to_collection(collection, docs_to_add)

    def _create_collection(self, collection: str):
        """Creates a new Weaviate collection with configured properties."""
        logger.info("Creating new Weaviate collection: %s", collection)

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
                "No metadata structure found for collection '%s' "
                "in config.yaml. No custom properties will be created.",
                collection
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to create collection: %s", e, exc_info=True)

    def _delete_documents(self, collection_obj, docs_to_delete: List[LoadedDocument]):
        """Deletes documents from the collection."""
        deleted_sources = [os.path.basename(doc.path) for doc in docs_to_delete]
        logger.info(
            "Removing %d deleted files from vector store: %s",
            len(deleted_sources),
            ', '.join(sorted(deleted_sources))
        )
        collection_obj.data.delete_many(
            where=wvc.query.Filter.by_property("source").contains_any(deleted_sources)
        )

    def _filter_upserts(
        self, collection_obj, docs_to_upsert: List[LoadedDocument]
    ) -> List[LoadedDocument]:
        """Filters documents that need to be updated and cleans up old versions."""
        existing_docs_map = {}

        # Batch the checking to avoid hitting limits with large number of chunks
        # and to handle potential truncation if a file has many chunks.
        batch_size = 10
        limit = 1000

        for i in range(0, len(docs_to_upsert), batch_size):
            batch = docs_to_upsert[i : i + batch_size]
            batch_sources = [os.path.basename(doc.path) for doc in batch]

            try:
                response = collection_obj.query.fetch_objects(
                    filters=wvc.query.Filter.by_property("source").contains_any(
                        batch_sources
                    ),
                    return_properties=["source", "last_modification"],
                    limit=limit
                )

                batch_map = {
                    obj.properties["source"]: obj.properties["last_modification"]
                    for obj in response.objects
                }
                existing_docs_map.update(batch_map)

                # If we hit the limit, some files might be missing due to truncation.
                # Check missing files individually to be safe.
                if len(response.objects) >= limit:
                    missing_sources = set(batch_sources) - set(batch_map.keys())
                    for source in missing_sources:
                        try:
                            single_res = collection_obj.query.fetch_objects(
                                filters=wvc.query.Filter.by_property("source").equal(source),
                                return_properties=["source", "last_modification"],
                                limit=1
                            )
                            if single_res.objects:
                                obj = single_res.objects[0]
                                existing_docs_map[obj.properties["source"]] = (
                                    obj.properties["last_modification"]
                                )
                        except Exception as e:
                            logger.warning("Error checking individual file %s: %s", source, e)

            except Exception as e:
                logger.warning("Error fetching existing docs for batch: %s", e)

        docs_that_need_update = []
        for doc in docs_to_upsert:
            source_name = os.path.basename(doc.path)
            if (source_name not in existing_docs_map or
                    doc.last_modified != existing_docs_map.get(source_name, 0)):
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
            collection_obj.data.delete_many(
                where=wvc.query.Filter.by_property("source").contains_any(sources_to_update)
            )
            return docs_that_need_update

        return []

    def _add_documents_to_collection(self, collection: str, docs_to_add: List[LoadedDocument]):
        """Prepares and adds documents to the collection."""
        location_config = next(
            (loc for loc in self._sync_locations
             if loc.collection == collection),
            None
        )

        documents_to_add = []
        for loaded_doc in docs_to_add:
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
                    for i, prop_setting in enumerate(location_config.metadata.structure):
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
                "Adding %d document chunks to the vector store...",
                len(docs)
            )
            try:
                ids = self._vectorstores[collection].add_documents(docs)
                logger.debug("Successfully added documents to the vector store %s", ids)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("An error occurred while adding documents: %s", e)
        else:
            logger.info("No documents to add to the vector store.")

    def start_periodic_sync(self):
        """Starts the background thread for periodic synchronization."""
        for loader in self._loaders:
            # Use functools.partial to create a callback with the collection
            # name pre-filled. The loader's watcher expects a callback that
            # only takes one argument (the documents).
            if loader['poll_interval'] == 0:
                logger.info(
                    "Skipping periodic synchronization watcher, collection: %s",
                    loader['collection']
                )
                continue
            update_callback = partial(
                self._update_vector_store,
                collection=loader['collection']
            )
            loader["loader"].watcher(callback=update_callback)
            logger.info(
                "Starting periodic synchronization watcher, collection: %s",
                loader['collection']
            )

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
        """Returns the name of the vector store manager."""
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
