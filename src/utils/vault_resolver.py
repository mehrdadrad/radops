import logging
import re
from typing import Any

import hvac

logger = logging.getLogger(__name__)


def _fetch_vault_secret(
    secret_path: str, secret_key: str | None, vault_client: hvac.Client, mount_point: str
) -> Any:
    try:
        secret = vault_client.secrets.kv.v2.read_secret_version(
            path=secret_path, 
            mount_point=mount_point,
            raise_on_deleted_version=True
        )

        if not (secret and "data" in secret and "data" in secret["data"]):
            logger.error(f"Secret not found or empty at path '{secret_path}'")
            return None

        secret_values = secret["data"]["data"]
        if secret_key:
            if secret_key in secret_values:
                return secret_values[secret_key]
            else:
                logger.error(
                    f"Key '{secret_key}' not found in secret at path '{secret_path}'"
                )
                return None
        else:
            if len(secret_values) == 1:
                return list(secret_values.values())[0]
            else:
                return secret_values
    except Exception as e:
        logger.error(
            f"Failed to resolve vault secret for path '{secret_path}': {e}"
        )
        return None


def _resolve_value(value: str, vault_client: hvac.Client, mount_point: str) -> Any:
    """
    Resolves 'vault:...' strings.
    The format is `vault:path/to/secret#key`.
    """
    if not isinstance(value, str):
        return value

    pattern = r"vault:([^#\s]+)(?:#(\S+))?"

    # Check for full match to preserve types (e.g. dict return)
    full_match = re.fullmatch(pattern, value)
    if full_match:
        secret_path = full_match.group(1)
        secret_key = full_match.group(2)
        result = _fetch_vault_secret(secret_path, secret_key, vault_client, mount_point)
        return result if result is not None else value

    # Check for partial matches for string interpolation
    if "vault:" in value:
        def replace(match):
            secret_path = match.group(1)
            secret_key = match.group(2)
            result = _fetch_vault_secret(secret_path, secret_key, vault_client, mount_point)
            if result is None:
                return match.group(0)
            return str(result)

        return re.sub(pattern, replace, value)

    return value


def resolve_vault_secrets(
    config: Any, vault_client: hvac.Client, mount_point: str
) -> Any:
    """
    Recursively traverses a config object (dict or list) and replaces vault
    paths with secrets.
    """
    if isinstance(config, dict):
        return {
            key: resolve_vault_secrets(value, vault_client, mount_point)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [
            resolve_vault_secrets(item, vault_client, mount_point) for item in config
        ]
    if isinstance(config, str):
        return _resolve_value(config, vault_client, mount_point)
    return config