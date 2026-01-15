"""Module for LLM factory and configuration."""
import logging
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.config import settings
from services.telemetry.telemetry import telemetry

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


class LLMCallbackHandler(BaseCallbackHandler):
    """Callback handler to log LLM errors and warnings."""

    def __init__(self, agent_name: str = None):
        self.agent_name = agent_name

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        logger.error("LLM Error detected: %s", error)
        attributes = {}
        if self.agent_name:
            attributes["agent"] = self.agent_name
        telemetry.update_counter("agent.llm.errors", attributes=attributes)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        logger.error("Tool Error detected: %s", error)
        attributes = {}
        if self.agent_name:
            attributes["agent"] = self.agent_name
        telemetry.update_counter("agent.tool.errors", attributes=attributes)

    def on_chain_error(self, error: BaseException, **kwargs):
        logger.error("Chain Error detected: %s", error)

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """Check for truncation when LLM finishes."""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            total_tokens = usage.get("total_tokens", 0)
            attributes = {}
            if self.agent_name:
                attributes["agent"] = self.agent_name

            if total_tokens > 0:
                telemetry.update_counter("agent.llm.tokens.total", total_tokens, attributes=attributes)

            # Prompt Caching Metrics
            cache_read = 0
            cache_creation = 0

            # Anthropic strategy
            if "cache_read_input_tokens" in usage:
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_creation = usage.get("cache_creation_input_tokens", 0)
            # OpenAI strategy
            elif "prompt_tokens_details" in usage:
                details = usage["prompt_tokens_details"]
                if isinstance(details, dict):
                    cache_read = details.get("cached_tokens", 0)

            if cache_read > 0:
                telemetry.update_counter("agent.llm.tokens.cache_read", cache_read, attributes=attributes)
            if cache_creation > 0:
                telemetry.update_counter("agent.llm.tokens.cache_creation", cache_creation, attributes=attributes)

        if hasattr(response, "generations"):
            for generations in response.generations:
                for gen in generations:
                    info = gen.generation_info or {}
                    reason = info.get("finish_reason")
                    if reason in ["length", "max_tokens"]:
                        logger.warning("LLM response reached max_tokens limit and was truncated.")

async def close_shared_client():
    """Closes the shared httpx.AsyncClient."""
    if not _shared_http_aclient.is_closed:
        try:
            await _shared_http_aclient.aclose()
            logger.info("Shared HTTP async client closed.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error closing shared HTTP async client: %s", e)


def llm_factory(profile_name: str, agent_name: str = None) -> BaseChatModel:
    """
    Factory function to get a configured LLM based on a profile.

    The configuration for profiles is expected to be in `settings.LLM.profiles`.
    Each profile should have a `provider` and its specific settings.
    """
    try:
        profile_settings = settings.llm.profiles[profile_name]
        provider = profile_settings.provider
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM profile: {profile_name}") from exc
    except AttributeError as exc:
        raise ValueError(f"Profile '{profile_name}' is missing a 'provider' attribute.") from exc

    temperature = (
        profile_settings.temperature
        if profile_settings.temperature is not None
        else 0.7
    )

    callbacks = [LLMCallbackHandler(agent_name=agent_name)]

    match provider:
        case "openai" | "deepseek":

            return ChatOpenAI(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                api_key=profile_settings.api_key,
                base_url=profile_settings.base_url,
                http_async_client=_shared_http_aclient,
                callbacks=callbacks,
            )
        case "anthropic":
            return ChatAnthropic(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                api_key=profile_settings.api_key,
                base_url=profile_settings.base_url,
                callbacks=callbacks,
            )
        case "azure":
            return AzureChatOpenAI(
                azure_deployment=profile_settings.model,
                openai_api_version=profile_settings.api_version,
                azure_endpoint=profile_settings.base_url,
                api_key=profile_settings.api_key,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                model_version=profile_settings.model_version,
                callbacks=callbacks,
            )
        case "gemini":
            return ChatGoogleGenerativeAI(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                google_api_key=profile_settings.api_key,
                project=profile_settings.google_project,
                location=profile_settings.google_location,
                callbacks=callbacks,
            )
        case "groq":
            return ChatGroq(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                api_key=profile_settings.api_key,
                reasoning_format=profile_settings.reasoning_format,
                callbacks=callbacks,
            )
        case "mistral":
            return ChatMistralAI(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                api_key=profile_settings.api_key,
                endpoint=profile_settings.base_url,
                callbacks=callbacks,
            )
        case "bedrock":
            return ChatBedrockConverse(
                model=profile_settings.model,
                temperature=temperature,
                max_tokens=profile_settings.max_tokens,
                region_name=profile_settings.aws_region,
                aws_access_key_id=profile_settings.aws_access_key_id,
                aws_secret_access_key=profile_settings.aws_secret_access_key,
                aws_session_token=profile_settings.aws_session_token,
                callbacks=callbacks,
            )

        case "ollama":
            return ChatOllama(
                model=profile_settings.model,
                temperature=temperature,
                base_url=profile_settings.base_url,
                callbacks=callbacks,
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
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM profile for embeddings: {profile_name}") from exc
    except AttributeError as exc:
        raise ValueError(f"Profile '{profile_name}' is missing a 'provider' attribute.") from exc

    match provider:
        case "openai" | "deepseek":
            return OpenAIEmbeddings(
                model=profile_settings.model,
                api_key=profile_settings.api_key,
                base_url=profile_settings.base_url,
                dimensions=profile_settings.dimensions,
            )
        case "azure":
            return AzureOpenAIEmbeddings(
                azure_deployment=profile_settings.model,
                openai_api_version=profile_settings.api_version,
                azure_endpoint=profile_settings.base_url,
                api_key=profile_settings.api_key,
            )
        case "gemini":
            return GoogleGenerativeAIEmbeddings(
                model=profile_settings.model,
                google_api_key=profile_settings.api_key,
            )
        case "bedrock":
            return BedrockEmbeddings(
                model_id=profile_settings.model,
                region_name=profile_settings.aws_region,
                aws_access_key_id=profile_settings.aws_access_key_id,
                aws_secret_access_key=profile_settings.aws_secret_access_key,
                aws_session_token=profile_settings.aws_session_token,
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
        