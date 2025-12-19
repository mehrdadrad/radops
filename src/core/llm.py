from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.config import settings


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

    match provider:
        case "openai" | "deepseek":
            temperature = (
                profile_settings.temperature
                if profile_settings.temperature is not None
                else 0.7
            )
            return ChatOpenAI(
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