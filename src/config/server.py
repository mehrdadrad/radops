"""Server configuration settings."""
import logging
import sys
from typing import Any, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.utils import get_config_path, load_yaml_config

logger = logging.getLogger(__name__)


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root.
    """
    config_path = get_config_path("server.yaml")
    return load_yaml_config(config_path)


class ServerSettings(BaseSettings):
    """
    Server settings.
    Settings are loaded from server.yaml and environment variables.
    """
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
            env_settings,
            lambda: yaml_config_settings_source(settings_cls),
        )

    host: str = "0.0.0.0"
    port: int = 8005
    plain_message: bool = False
    skip_initial_sync: bool = False
    auth_disabled: bool = False
    service_api_key: Optional[str] = Field(default=None, validation_alias="RADOPS_SERVICE_API_KEY")
    service_token: Optional[str] = Field(default=None, validation_alias="RADOPS_SERVICE_TOKEN")


# Instantiate the settings
try:
    server_settings = ServerSettings()
except ValidationError as e:
    print(f"Invalid server configuration: {e}", file=sys.stderr)
    sys.exit(1)
