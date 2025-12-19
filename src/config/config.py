import logging
import os
from typing import Any, Dict, Optional

import hvac
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from libs.vault_resolver import resolve_vault_secrets

logger = logging.getLogger(__name__)


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root and resolves any vault secrets.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'
    )

    try:
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

    # Resolve vault secrets
    vault_config = yaml_data.get("vault", {})
    vault_url = os.environ.get("VAULT_URL", vault_config.get("url"))
    vault_token = os.environ.get("VAULT_TOKEN", vault_config.get("token"))

    if vault_url and vault_token:
        vault_mount_point = os.environ.get(
            "VAULT_MOUNT_POINT", vault_config.get("mount_point", "secret")
        )
        try:
            vault_client = hvac.Client(url=vault_url, token=vault_token)
            if vault_client.is_authenticated():
                return resolve_vault_secrets(
                    yaml_data, vault_client, vault_mount_point
                )
            else:
                logger.warning(
                    "Vault authentication failed. Skipping secret resolution from config.yaml."
                )
        except Exception as e:
            logger.error(
                f"Error connecting to Vault or resolving secrets from config.yaml: {e}"
            )

    return yaml_data


class LLMProfileSettings(BaseModel):
    """Settings for a single LLM profile."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class LLMSettings(BaseModel):
    """Settings for all LLM profiles."""
    default_profile: str = "openai-main"
    profiles: Dict[str, LLMProfileSettings] = Field(default_factory=dict)


class SummarizationSettings(BaseModel):
    keep_message: int = 50
    llm_profile: str = "openai-summary"


class TTLSettings(BaseModel):
    """Settings for Redis TTL."""
    time_minutes: int
    refresh_on_read: bool


class RedisSettings(BaseModel):
    """Settings for Redis memory backend."""
    endpoint: str
    ttl: TTLSettings


class MemorySettings(BaseModel):
    """Settings for conversation memory."""
    redis: RedisSettings
    summarization: SummarizationSettings = Field(
        default_factory=SummarizationSettings
    )


class Mem0VectorStoreConfigSettings(BaseModel):
    """Settings for mem0 vector store config."""
    path: Optional[str] = None
    collection_name: Optional[str] = None
    cluster_url: Optional[str] = None


class Mem0VectorStoreSettings(BaseModel):
    """Settings for mem0 vector store."""
    provider: str
    config: Mem0VectorStoreConfigSettings


class Mem0Settings(BaseModel):
    llm_profile: str
    embedding_profile: str
    vector_store: Mem0VectorStoreSettings
    excluded_tools: list[str] = Field(default_factory=list)


class GuardrailSettings(BaseModel):
    enabled: bool = False


class MetadataStructureSetting(BaseModel):
    name: str
    description: str


class MetadataSettings(BaseModel):
    delimiter: Optional[str] = None
    structure: list[MetadataStructureSetting] = Field(default_factory=list)


class SyncLocationSettings(BaseModel):
    """Settings for a single sync location."""
    name: str
    type: str
    path: str
    collection: str
    sync_interval: int
    prompt_file: Optional[str] = None
    prompt: Optional[str] = None
    metadata: Optional[MetadataSettings] = None


class VectorStoreProfileSettings(BaseModel):
    """Settings for a single vector store profile."""
    name: str
    provider: str
    embedding_profile: str
    sync_locations: list[SyncLocationSettings] = Field(default_factory=list)


class VectorStoreSettings(BaseModel):
    """Settings for the vector store."""
    profiles: list[VectorStoreProfileSettings] = Field(default_factory=list)
    providers: Dict[str, Any] = Field(default_factory=dict)


class LogSettings(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None
    retention: str = "1 week"
    rotation: str = "10 MB"


class VaultSettings(BaseModel):
    url: Optional[str] = None
    token: Optional[str] = None
    mount_point: str = "secret"


class Settings(BaseSettings):
    """
    Centralized application settings.
    Settings are loaded from config.yaml and environment variables.
    """
    # model_config for pydantic-settings
    model_config = SettingsConfigDict(extra='ignore', env_nested_delimiter='__')

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            lambda: yaml_config_settings_source(settings_cls),
            env_settings,
        )

    logging: LogSettings = Field(default_factory=LogSettings)

    # LLM
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Vector Store
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

    guardrail: GuardrailSettings = Field(default_factory=GuardrailSettings)

    # Memory
    memory: MemorySettings = Field(default_factory=MemorySettings)

    # Mem0
    mem0: Mem0Settings = Field(default_factory=Mem0Settings)

    # Vault
    vault: VaultSettings = Field(default_factory=VaultSettings)

    # OpenTelemetry
    opentelemetry: Dict[str, Any] = Field(default_factory=dict)

# Instantiate the settings
settings = Settings()