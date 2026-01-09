"""Handles settings for various tools integrated with the application."""
import logging
import os
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.config import settings as app_settings
from config.utils import get_config_path, load_yaml_config, process_vault_secrets

logger = logging.getLogger(__name__)


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root and resolves any vault secrets.
    """
    config_path = get_config_path("tools.yaml")

    yaml_data = load_yaml_config(config_path)

    # Resolve vault secrets
    vault_url = os.environ.get('VAULT_URL', app_settings.vault.url)
    vault_token = os.environ.get('VAULT_TOKEN', app_settings.vault.token)
    vault_mount_point = os.environ.get(
        'VAULT_MOUNT_POINT', app_settings.vault.mount_point
    )

    return process_vault_secrets(
        yaml_data, vault_url, vault_token, vault_mount_point, "tools.yaml"
    )

class ToolFunctionConfig(BaseModel):
    """Configuration for a specific tool function."""
    function: str
    enabled: bool = True

class LocalToolConfig(BaseModel):
    """Configuration for local tools loaded from a module."""
    module: str
    tools: list[ToolFunctionConfig] = Field(default_factory=list)

class GithubSettings(BaseModel):
    """Settings for GitHub integration."""

    server: Optional[str] = None
    default_org: Optional[str] = None
    default_repo: Optional[str] = None


class JiraSettings(BaseModel):
    """Settings for Jira integration."""

    server: Optional[str] = None


class PeeringDBSettings(BaseModel):
    """Settings for PeeringDB integration."""

    api_key: str = None


class GeoIPSettings(BaseModel):
    """Settings for GeoIP integration."""

    database_path: str = None


class AWSSettings(BaseModel):
    """Settings for AWS integration."""

    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: Optional[str] = None

class MCPSettings(BaseModel):
    """Settings for MCP integration."""
    servers: Dict[str, Any] = Field(default_factory=dict)
    retry_attempts: int = 3
    retry_delay: int = 5
    persistent_interval: int = 60
    execution_timeout: float = 60.0
    connect_timeout: float = 10.0
    health_check_interval: float = 10.0


class ToolSettings(BaseSettings):
    """
    Centralized tools settings.
    Settings are loaded from config_tools.yaml and environment variables.
    """

    # model_config for pydantic-settings
    model_config = SettingsConfigDict(extra='allow', env_nested_delimiter='__')

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

    # Local Tools
    local_tools: list[LocalToolConfig] = Field(default_factory=list)

    # MCP Client Configuration
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    # PeeringDB
    peeringdb: PeeringDBSettings = Field(default_factory=PeeringDBSettings)

    # Github
    github: GithubSettings = Field(default_factory=GithubSettings)

    # Jira
    jira: JiraSettings = Field(default_factory=JiraSettings, validation_alias='jira')

    # GeoIP
    geoip: GeoIPSettings = Field(default_factory=GeoIPSettings)

    # AWS
    aws: AWSSettings = Field(default_factory=AWSSettings)


# Instantiate the settings
try:
    tool_settings = ToolSettings()
except ValidationError as e:
    print("The application failed to start because of invalid configuration in 'tools.yaml'.\n", file=sys.stderr)

    for error in e.errors():
        field_path = " -> ".join(str(x) for x in error['loc'])
        message = error['msg']
        print(f"  • \033[1m{field_path}\033[0m: {message}", file=sys.stderr)

    print("\nPlease verify your 'tools.yaml' file matches the expected structure.", file=sys.stderr)
    print("For detailed instructions, please refer to 'docs/tools_guide.md'.", file=sys.stderr)
    sys.exit(1)
