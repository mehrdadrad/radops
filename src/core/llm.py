import asyncio
import atexit
import logging

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.config import settings

logger = logging.getLogger(__name__)

limits = httpx.Limits(
    # How many connections to keep open in the pool.
    max_keepalive_connections=20,
    # Total max connections allowed.
    max_connections=100,
    # CRITICAL: Default is only 5 seconds!
    # If the user takes 10s to type a reply, the connection drops.
    # Increase this to keep the TCP pipe hot while the user thinks.
    keepalive_expiry=60.0,
)
_shared_http_aclient = httpx.AsyncClient(limits=limits)


def _close_shared_http_client():
    """Closes the shared httpx.AsyncClient on application exit."""
    if not _shared_http_aclient.is_closed:
        try:
            # Use asyncio.run() to execute the async close method
            # from the synchronous context of atexit.
            asyncio.run(_shared_http_aclient.aclose())
            logger.info("Shared HTTP async client closed.")
        except Exception as e:
            logger.error(f"Error closing shared HTTP async client: {e}")


atexit.register(_close_shared_http_client)


def llm_factory(profile_name: str) -> BaseChatModel:
    """
    Factory function to get a configured LLM based on a profile.

    The configuration for profiles is expected to be in `settings.LLM.profiles`.
    Each profile should have a `provider` and its specific settings.
    """
    try:
        profile_settings = settings.llm.profiles[profile_name]
        provider = profile_settings.provider
    except KeyError:
        raise ValueError(f"Unsupported LLM profile: {profile_name}")
    except AttributeError:
        raise ValueError(f"Profile '{profile_name}' is missing a 'provider' attribute.")

    temperature = (
        profile_settings.temperature
        if profile_settings.temperature is not None
        else 0.7
    )

    match provider:
        case "openai" | "deepseek":

            return ChatOpenAI(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                api_key=profile_settings.api_key,
                base_url=profile_settings.base_url,
                http_async_client=_shared_http_aclient,
            )
        case "anthropic":
            return ChatAnthropic(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                api_key=profile_settings.api_key,
                base_url=profile_settings.base_url,
            )
        case "ollama":
            return ChatOllama(
                model=profile_settings.model,
                base_url=profile_settings.base_url,
            )
        case _:
            raise ValueError(
                f"Unsupported LLM provider in profile '{profile_name}': "
                f"{provider}"
            )


def embedding_factory(profile_name: str) -> Embeddings:
    """
    Factory function to get a configured embedding model based on a profile.

    The configuration for profiles is expected to be in `settings.llm.profiles`.
    Each profile should have a `provider` and its specific settings.
    """
    try:
        profile_settings = settings.llm.profiles[profile_name]
        provider = profile_settings.provider
    except KeyError:
        raise ValueError(f"Unsupported LLM profile for embeddings: {profile_name}")
    except AttributeError:
        raise ValueError(f"Profile '{profile_name}' is missing a 'provider' attribute.")

    match provider:
        case "openai" | "deepseek":
            return OpenAIEmbeddings(
                model=profile_settings.model,
                api_key=profile_settings.api_key,
                base_url=profile_settings.base_url,
            )
        case "ollama":
            return OllamaEmbeddings(
                model=profile_settings.model,
                base_url=profile_settings.base_url
            )
        case _:
            raise ValueError(
                f"Unsupported embedding provider in profile '{profile_name}': "
                f"{provider}"
            )