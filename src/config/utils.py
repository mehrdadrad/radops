"""Shared configuration utilities."""
import logging
from typing import Any, Dict, Optional

import hvac
import yaml
from pydantic_settings import BaseSettings

from libs.vault_resolver import resolve_vault_secrets

logger = logging.getLogger(__name__)


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
