"""Module for managing graph checkpointers (persistence)."""
import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from redis.asyncio import Redis as AsyncRedis

from config.config import settings


@asynccontextmanager
async def get_checkpointer():
    """
    Context manager that yields a checkpointer (Redis or in-memory).
    Handles setup and potential connection errors.
    """
    if settings.memory.redis.endpoint:
        ttl_config = None
        if settings.memory.redis.ttl:
            ttl_config = {
                "default_ttl": settings.memory.redis.ttl.time_minutes,
                "refresh_on_read": settings.memory.redis.ttl.refresh_on_read,
            }

        logging.info("Persistence is enabled. Connecting to Redis ...")
        logging.info("  - %s", settings.memory.redis.endpoint)
        logging.info("  - TTL config: %s", ttl_config)

        redis_client = None
        try:
            redis_client = AsyncRedis.from_url(settings.memory.redis.endpoint)
            async with AsyncRedisSaver(
                redis_client=redis_client, ttl=ttl_config
            ) as checkpointer:
                await checkpointer.asetup()
                yield checkpointer, redis_client
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning(
                "Redis connection failed: %s. "
                "Falling back to in-memory persistence.",
                e
            )
            yield MemorySaver(), None
    else:
        logging.info("Persistence is disabled. Running in-memory.")
        yield MemorySaver(), None
