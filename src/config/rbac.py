"""Handles Role-Based Access Control (RBAC) settings for the application."""
import os
import sys
from typing import Any, Dict

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from config.utils import get_config_path, load_yaml_config


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root.
    """
    config_path = get_config_path("rbac.yaml")
    return load_yaml_config(config_path)


class RBACSettings(BaseSettings):
    """
    Centralized RBAC settings.
    Settings are loaded from rbac.yaml and environment variables.
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

    # RBAC
    users: Dict[str, str] = Field(default_factory=dict)
    role_permissions: Dict[str, list[str]] = Field(default_factory=dict)

# Instantiate the settings
try:
    rbac_settings = RBACSettings()
except ValidationError as e:
    print("The application failed to start because of invalid configuration in 'rbac.yaml'.\n", file=sys.stderr)

    for error in e.errors():
        field_path = " -> ".join(str(x) for x in error['loc'])
        message = error['msg']
        print(f"  • \033[1m{field_path}\033[0m: {message}", file=sys.stderr)

    print("\nPlease verify your 'rbac.yaml' file matches the expected structure.", file=sys.stderr)
    print("For detailed instructions, please refer to 'docs/rbac_guide.md'.", file=sys.stderr)
    sys.exit(1)