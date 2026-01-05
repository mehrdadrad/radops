"""Handles integration settings for the application."""
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
    config_path = get_config_path("integrations.yaml")

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

class GithubProfileSettings(BaseModel):
    """Settings for a single GitHub integration profile."""
    token: str = None
    base_url: Optional[str] = None

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
    # Github
    github: Dict[str, GithubProfileSettings] = Field(default_factory=dict)


# Instantiate the settings
try:
    integration_settings = IntegrationSettings()
except ValidationError as e:
    print("The application failed to start because of invalid configuration in 'integrations.yaml'.\n", file=sys.stderr)

    for error in e.errors():
        field_path = " -> ".join(str(x) for x in error['loc'])
        message = error['msg']
        print(f"  • \033[1m{field_path}\033[0m: {message}", file=sys.stderr)

    print("\nPlease verify your 'integrations.yaml' file matches the expected structure.", file=sys.stderr)
    print("For detailed instructions, please refer to 'docs/integrations_guide.md'.", file=sys.stderr)
    sys.exit(1)