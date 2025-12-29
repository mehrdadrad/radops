"""This module provides a factory for creating and configuring vector store managers."""
import atexit
import logging

from config.config import settings
from core.llm import embedding_factory
from storage.chromadb import ChromaVectorStoreManager
from storage.protocols import VectorStoreManager
from storage.pinecone import PineconeVectorStoreManager
from storage.qdrant import QdrantVectorStoreManager
from storage.weaviatedb import WeaviateVectorStoreManager

logger = logging.getLogger(__name__)

def vector_store_factory() -> list[VectorStoreManager]:
    """Creates and configures vector store managers based on settings.

    This factory reads the `settings.vector_store.profiles` to determine which
    vector store providers (e.g., Chroma, Weaviate) to initialize. It configures
    each manager with the appropriate embedding model and sync locations.

    An `atexit` handler is registered for each manager to ensure proper
    cleanup and connection closing on application exit.

    Returns:
        A list of initialized VectorStoreManager instances.

    Raises:
        ValueError: If an unsupported vector store provider is specified.
    """
    managers = []

    for profile in settings.vector_store.profiles:  # pylint: disable=no-member
        provider = profile.provider
        embedding_profile = profile.embedding_profile
        sync_locations = profile.sync_locations

        logger.info("Vector store provider: '%s'", provider)
        logger.info("Embedding profile: '%s'", embedding_profile)
        logger.info("Sync locations: %s", [loc.name for loc in sync_locations])

        embeddings_model = embedding_factory(embedding_profile)

        if provider == "chroma":
            manager = ChromaVectorStoreManager(
                profile.name, sync_locations, embeddings_model
            )
        elif provider == "weaviate":
            manager = WeaviateVectorStoreManager(
                profile.name, sync_locations, embeddings_model
            )  # type: ignore
        elif provider == "pinecone":
            manager = PineconeVectorStoreManager(
                profile.name, sync_locations, embeddings_model
            )
        elif provider == "qdrant":
            manager = QdrantVectorStoreManager(
                profile.name, sync_locations, embeddings_model
            )
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

        atexit.register(manager.close)
        managers.append(manager)

    return managers
