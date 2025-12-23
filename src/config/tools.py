"""Handles settings for various tools integrated with the application."""
import logging
import os
from typing import Any, Dict, Optional

import hvac
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.config import settings as app_settings
from libs.vault_resolver import resolve_vault_secrets

logger = logging.getLogger(__name__)


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root and resolves any vault secrets.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'config', 'tools.yaml'
    )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # Return empty dict if config_tools.yaml is not found
        return {}

    # Resolve vault secrets
    vault_url = os.environ.get('VAULT_URL', app_settings.vault.url)
    vault_token = os.environ.get('VAULT_TOKEN', app_settings.vault.token)

    if vault_url and vault_token:
        vault_mount_point = os.environ.get(
            'VAULT_MOUNT_POINT', app_settings.vault.mount_point
        )
        try:
            vault_client = hvac.Client(url=vault_url, token=vault_token)
            if vault_client.is_authenticated():
                return resolve_vault_secrets(
                    yaml_data, vault_client, vault_mount_point
                )
            else:
                logger.warning(
                    'Vault authentication failed. Skipping secret resolution from tools.yaml.'
                )
        except Exception as e:
            logger.error(
                f'Error connecting to Vault or resolving secrets from tools.yaml: {e}'
            )

    return yaml_data


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


class ToolSettings(BaseSettings):
    """
    Centralized tools settings.
    Settings are loaded from config_tools.yaml and environment variables.
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

    # MCP Client Configuration
    mcp_servers: Optional[Dict[str, Any]] = None

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
tool_settings = ToolSettings()
