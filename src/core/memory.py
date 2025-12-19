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
            if mem0_config.vector_store.provider == "chroma":
                if mem0_config.vector_store.config.path:
                    vector_store_config["path"] = (
                        mem0_config.vector_store.config.path
                    )
            elif mem0_config.vector_store.provider == "weaviate":
                if mem0_config.vector_store.config.collection_name:
                    vector_store_config["collection_name"] = (
                        mem0_config.vector_store.config.collection_name
                    )
                    vector_store_config["cluster_url"] = (
                        mem0_config.vector_store.config.cluster_url
                    )

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
        if (self._mem0_client and
                hasattr(self._mem0_client, 'vector_store') and
                hasattr(self._mem0_client.vector_store, 'client')):
            if hasattr(self._mem0_client.vector_store.client, 'close'):
                # Some clients might not have a close method,
                # or it might not be async
                try:
                    client = self._mem0_client.vector_store.client
                    if asyncio.iscoroutinefunction(client.close):
                        await client.close()
                    else:
                        client.close()
                    logger.info("Mem0 client closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing mem0 client: {e}")


# Singleton instance
mem0_manager = Mem0Manager()


async def get_mem0_client():
    return await mem0_manager.get_client()
