"""Handles integration settings for the application."""
import logging
import os
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.config import settings as app_settings
from config.utils import load_yaml_config, process_vault_secrets

logger = logging.getLogger(__name__)


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root and resolves any vault secrets.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'config', 'integrations.yaml'
    )

    yaml_data = load_yaml_config(config_path)

    # Resolve vault secrets
    vault_url = os.environ.get("VAULT_URL", app_settings.vault.url)
    vault_token = os.environ.get("VAULT_TOKEN", app_settings.vault.token)
    vault_mount_point = os.environ.get(
        "VAULT_MOUNT_POINT", app_settings.vault.mount_point
    )

    return process_vault_secrets(
        yaml_data, vault_url, vault_token, vault_mount_point, "integrations.yaml"
    )


class SlackSettings(BaseModel):
    """Settings for Slack integration."""
    bot_token: Optional[str] = None
    app_token: Optional[str] = None
    log_level: str = "INFO"

class IntegrationSettings(BaseSettings):
    """
    Centralized integrations settings.
    Settings are loaded from integrations.yaml and environment variables.
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

    # Slack
    slack: SlackSettings = Field(default_factory=SlackSettings)


# Instantiate the settings
integration_settings = IntegrationSettings()