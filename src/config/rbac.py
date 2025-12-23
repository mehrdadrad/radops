"""Handles Role-Based Access Control (RBAC) settings for the application."""
import os
from typing import Any, Dict

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', '..','config', 'rbac.yaml')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        # Return empty dict if rbac.yaml is not found
        return {}


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
rbac_settings = RBACSettings()