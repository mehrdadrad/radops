"""Shared configuration utilities."""
import logging
import os
import sys
from typing import Any, Dict, Optional

import hvac
import yaml
from pydantic_settings import BaseSettings

from libs.vault_resolver import resolve_vault_secrets

logger = logging.getLogger(__name__)


def get_config_path(filename: str) -> str:
    """Resolves the path to a configuration file."""
    base_dir = os.environ.get("RADOPS_CONFIG_DIR")
    if not base_dir:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")

    config_path = os.path.abspath(os.path.join(base_dir, filename))
    if not os.path.exists(config_path):
        logger.error(
            "Configuration file not found: %s. "
            "You can set the configuration directory using the "
            "RADOPS_CONFIG_DIR environment variable.",
            config_path,
        )
        sys.exit(1)
    return config_path


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def process_vault_secrets(
    yaml_data: Dict[str, Any],
    vault_url: Optional[str],
    vault_token: Optional[str],
    vault_mount_point: Optional[str],
    source_name: str,
) -> Dict[str, Any]:
    """Resolves Vault secrets in the configuration data."""
    if not (vault_url and vault_token):
        return yaml_data

    try:
        vault_client = hvac.Client(url=vault_url, token=vault_token)
        if vault_client.is_authenticated():
            return resolve_vault_secrets(
                yaml_data, vault_client, vault_mount_point
            )
        logger.warning(
            "Vault authentication failed. "
            "Skipping secret resolution from %s.",
            source_name,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Error connecting to Vault or resolving secrets from %s: %s",
            source_name,
            e,
        )

    return yaml_data
