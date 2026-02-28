"""
This module defines the configuration for the RadOps assistant.
"""
import logging
import os
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.utils import get_config_path, load_yaml_config, process_vault_secrets

logger = logging.getLogger(__name__)


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root and resolves any vault secrets.
    """
    config_path = get_config_path("config.yaml")

    yaml_data = load_yaml_config(config_path)

    # Resolve vault secrets
    vault_config = yaml_data.get("vault", {})
    vault_url = os.environ.get("VAULT_URL", vault_config.get("url"))
    vault_token = os.environ.get("VAULT_TOKEN", vault_config.get("token"))
    vault_mount_point = os.environ.get(
        "VAULT_MOUNT_POINT", vault_config.get("mount_point", "secret")
    )
    return process_vault_secrets(
        yaml_data, vault_url, vault_token, vault_mount_point, "config.yaml"
    )


class LLMProfileSettings(BaseModel):
    """Settings for a single LLM profile."""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    model_version: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    dimensions: Optional[int] = None
    reasoning_format: Optional[str] = None
    
    aws_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    google_project: Optional[str] = None
    google_location: Optional[str] = None


class LLMSettings(BaseModel):
    """Settings for all LLM profiles."""

    default_profile: str
    profiles: Dict[str, LLMProfileSettings] = Field(default_factory=dict)


class AgentSettings(BaseModel):
    """Settings for agent configurations."""

    description: str = None
    llm_profile: str = None
    manifest_llm_profile: Optional[str] = None
    allow_tools: list[str] = Field(default_factory=list)
    allow_skills: list[str] = Field(default_factory=list)
    system_prompt_file: str


class GuardrailsSettings(BaseModel):
    """Settings for guardrails."""

    enabled: bool = False
    llm_profile: Optional[str] = None
    prompt_file: Optional[str] = None

    @model_validator(mode="after")
    def check_required_fields(self):
        """Validates that required fields are present when enabled."""
        if self.enabled:
            if not self.llm_profile:
                raise ValueError("llm_profile is required when guardrails are enabled")
            if not self.prompt_file:
                raise ValueError("prompt_file is required when guardrails are enabled")
        return self


class SupervisorSettings(BaseModel):
    """Settings for the supervisor agent."""

    llm_profile: Optional[str] = None
    prompt_file: Optional[str] = None
    discovery_mode: str = "prompt"
    discovery_threshold: float = 1.6

class SystemSettings(BaseModel):
    """Settings for the system agent."""

    llm_profile: Optional[str] = None


class AuditorSettings(BaseModel):
    """Settings for auditor."""

    enabled: bool = False
    llm_profile: str = None
    prompt_file: str = None
    threshold: float = 0.8


class DiscoverySettings(BaseModel):
    """Settings for discovery mechanisms (agents, skills)."""

    embedding_profile: str


class AgentsSettings(BaseModel):
    """Settings for all agents."""

    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    auditor: AuditorSettings = Field(default_factory=AuditorSettings)
    profiles: Dict[str, AgentSettings] = Field(default_factory=dict)


class SummarizationSettings(BaseModel):
    """Settings for memory summarization."""

    keep_message: int = 50
    token_threshold: int = 2000
    llm_profile: Optional[str] = None


class TTLSettings(BaseModel):
    """Settings for Redis TTL."""

    time_minutes: int
    refresh_on_read: bool


class ShortTermMemoryConfig(BaseModel):
    """Configuration for short-term memory backend."""
    url: str
    ttl: Optional[TTLSettings] = None


class ShortTermMemorySettings(BaseModel):
    """Settings for short-term memory."""
    provider: str = "redis"
    config: ShortTermMemoryConfig
    summarization: SummarizationSettings = Field(default_factory=SummarizationSettings)


class LongTermMemoryBackendConfig(BaseModel):
    """Settings for mem0 vector store config."""
    path: Optional[str] = None
    collection_name: Optional[str] = None
    cluster_url: Optional[str] = None
    api_key: Optional[str] = None
    index_name: Optional[str] = None
    environment: Optional[str] = None
    url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None


class LongTermMemoryConfig(BaseModel):
    """Settings for mem0 general config."""
    llm_profile: str
    embedding_profile: str
    limit: int = 10
    excluded_tools: list[str] = Field(default_factory=list)


class LongTermMemorySettings(BaseModel):
    """Settings for long-term memory."""
    provider: str = "mem0"
    backend: str
    config: LongTermMemoryConfig
    backend_config: LongTermMemoryBackendConfig


class MemorySettings(BaseModel):
    """Settings for conversation memory."""
    short_term: Optional[ShortTermMemorySettings] = None
    long_term: Optional[LongTermMemorySettings] = None

class GraphSettings(BaseModel):
    """Settings for graph execution."""

    max_concurrency: int = 5
    recursion_limit: int = 40


class MetadataStructureSetting(BaseModel):
    """Settings for metadata structure."""

    name: str
    description: str


class MetadataSettings(BaseModel):
    """Settings for metadata."""

    delimiter: Optional[str] = '_'
    structure: list[MetadataStructureSetting] = Field(default_factory=list)


class SyncLocationSettings(BaseModel):
    """Settings for a single sync location."""

    name: str
    type: str
    path: str
    collection: str
    sync_interval: Optional[int] = None
    prompt_file: Optional[str] = None
    prompt: Optional[str] = None
    metadata: Optional[MetadataSettings] = None
    loader_config: Optional[Dict[str, Any]] = None
    retrieval_config: Optional[Dict[str, Any]] = None


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
    """Settings for logging."""

    level: str = "INFO"
    file: Optional[str] = None
    retention: str = "1 week"
    rotation: str = "10 MB"


class VaultSettings(BaseModel):
    """Settings for Vault."""

    url: Optional[str] = None
    token: Optional[str] = None
    mount_point: str = "secret"


class LearningSettings(BaseModel):
    """Settings for adaptive learning."""

    enabled: bool = False
    dataset_path: str = "data/fine_tuning_dataset.jsonl"


class Settings(BaseSettings):
    """
    Centralized application settings.
    Settings are loaded from config.yaml and environment variables.
    """

    # model_config for pydantic-settings
    model_config = SettingsConfigDict(
        extra="ignore", env_nested_delimiter="__"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Customize the settings sources.
        """
        return (
            init_settings,
            lambda: yaml_config_settings_source(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    logging: LogSettings = Field(default_factory=LogSettings)

    # Agent
    agent: AgentsSettings = Field(default_factory=AgentsSettings)

    # LLM
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Discovery (agent and skills registry)
    discovery: DiscoverySettings

    # Vector Store
    vector_store: VectorStoreSettings = Field(
        default_factory=VectorStoreSettings
    )

    # Graph
    graph: GraphSettings = Field(default_factory=GraphSettings)

    # Memory
    memory: MemorySettings = Field(default_factory=MemorySettings)

    # Vault
    vault: VaultSettings = Field(default_factory=VaultSettings)

    # Learning
    learning: LearningSettings = Field(default_factory=LearningSettings)

    # Observability
    observability: Dict[str, Any] = Field(default_factory=dict)


# Instantiate the settings
try:
    settings = Settings()
except ValidationError as e:
    print("The application failed to start because of invalid configuration.\n", file=sys.stderr)

    for error in e.errors():
        field_path = " -> ".join(str(x) for x in error['loc'])
        message = error['msg']
        print(f"  • \033[1m{field_path}\033[0m: {message}", file=sys.stderr)

    print(
        "\nPlease verify your 'config.yaml' file matches the expected structure.",
        file=sys.stderr
    )
    print("For detailed instructions, please refer to 'docs/config_guide.md'.", file=sys.stderr)
    sys.exit(1)