import logging

import hvac

from config.config import settings

logger = logging.getLogger(__name__)

vault_client = None
try:
    if settings.vault and settings.vault.url and settings.vault.token:
        vault_client = hvac.Client(
            url=settings.vault.url,
            token=settings.vault.token,
        )
        logger.info("Vault client initialized for secret management.")
    else:
        logger.warning(
            "Vault settings (URL or Token) not found. "
            "Secret management tools will be disabled."
        )
except Exception as e:
    logger.error(
        f"Error initializing Vault client. Please check your Vault settings. "
        f"Error: {e}"
    )


def get_user_secrets(
    user_id: str, service: str, raise_on_error: bool = False
) -> dict:
    """
    Retrieves all secrets for a given user and service (e.g., 'github', 'jira')
    from the secure Vault store.
    """
    if not vault_client:
        error_message = (
            f"Error: Vault client is not initialized. "
            f"Cannot retrieve {service} token."
        )
        if raise_on_error:
            raise Exception(error_message)
        return {"error": error_message}

    secret_path = f"users/{user_id}/{service.lower()}"

    try:
        read_response = vault_client.secrets.kv.v2.read_secret_version(
            path=secret_path,
            mount_point=settings.vault.mount_point,
            raise_on_deleted_version=True
        )
        credentials = read_response.get("data", {}).get("data", {})

        return credentials or {
            "error": (
                f"No secrets found for service '{service}' "
                f"for user {user_id}."
            )
        }
    except hvac.exceptions.InvalidPath:
        error_message = (
            f"No secrets found for user {user_id}. "
            "Please set a token first using the set_app_token tool."
        )
    except Exception as e:
        error_message = (
            f"An error occurred while retrieving the {service} token "
            f"from Vault: {e}"
        )
        logger.error(error_message)

    if raise_on_error:
        raise Exception(error_message)
    return {"error": error_message}