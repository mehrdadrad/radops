import asyncio
import logging

from mem0 import AsyncMemory

from config.config import settings

logger = logging.getLogger(__name__)


class Mem0Manager:
    _instance = None
    _mem0_client: AsyncMemory | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Mem0Manager, cls).__new__(cls)
        return cls._instance

    async def get_client(self) -> AsyncMemory:
        if self._mem0_client is None:
            mem0_config = settings.mem0
            llm_profile_name = mem0_config.llm_profile
            llm_profile = settings.llm.profiles[llm_profile_name]
            embedder_profile_name = mem0_config.embedding_profile
            embedder_profile = settings.llm.profiles[embedder_profile_name]

            # Build vector store config dynamically based on provider
            vector_store_config = {}
            provider_fields_map = {
                "chroma": ["path"],
                "weaviate": ["collection_name", "cluster_url"],
                "pinecone": ["api_key", "index_name", "environment"],
                "qdrant": ["collection_name", "url", "host", "port", "api_key", "path"],
            }
            for field in provider_fields_map.get(mem0_config.vector_store.provider, []):
                if value := getattr(mem0_config.vector_store.config, field, None):
                    vector_store_config[field] = value

            # Construct the configuration for mem0's Memory.from_config
            config = {
                "vector_store": {
                    "provider": mem0_config.vector_store.provider,
                    "config": vector_store_config
                },
                "llm": {
                    "provider": llm_profile.provider,
                    "config": {
                        "model": llm_profile.model,
                        "api_key": llm_profile.api_key,
                        "openai_base_url": llm_profile.base_url
                    }
                },
                "embedder": {
                    "provider": embedder_profile.provider,
                    "config": {
                        "model": embedder_profile.model,
                        "api_key": embedder_profile.api_key,
                        "openai_base_url": embedder_profile.base_url
                    }
                }
            }
            self.__class__._mem0_client = await AsyncMemory.from_config(config)
        return self._mem0_client

    async def close(self):
        """
        Closes the underlying vector store client managed by mem0.
        This is important for providers like Weaviate to close sessions.
        Closes the mem0 client and its underlying connections.
        """
        if not self._mem0_client:
            return
        
        vector_store = getattr(self._mem0_client, "vector_store", None)
        client = getattr(vector_store, "client", None) if vector_store else None
        close_method = getattr(client, "close", None) if client else None

        if callable(close_method):
            try:
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
                logger.info("Mem0 client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing mem0 client: {e}")


# Singleton instance
mem0_manager = Mem0Manager()


async def get_mem0_client():
    return await mem0_manager.get_client()
