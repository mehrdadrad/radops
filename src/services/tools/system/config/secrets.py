import logging
from typing import Annotated, Optional

import hvac
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from config.config import settings
from libs.secrets import vault_client


class SetSecretsInput(BaseModel):
    user_id: Annotated[str, InjectedState("user_id")] = Field(
        description="The unique identifier for the current user."
    )
    service: str = Field(
        description=(
            "The service to save secrets for (e.g. 'github', 'jira'). "
            "If the user has not specified a service, you MUST ask them to "
            "provide one."
        )
    )
    token: Optional[str] = Field(description="The token for the service")
    username: Optional[str] = Field(
        description="The username for the Jira account. Required for 'jira' service."
    )
    github_org: Optional[str] = Field(
        description=(
            "The organization name of the GitHub repository (user or organization). "
            "Required for 'github' service."
        )
    )
    github_repo: Optional[str] = Field(
        description="The name of the GitHub repository. Required for 'github' service."
    )


logger = logging.getLogger(__name__)


@tool(args_schema=SetSecretsInput)
def secret__set_user_secrets(user_id: str, service: str, **details) -> str:
    """
    Saves or updates an API token and other details for a given user and
    service (e.g., 'github', 'jira') in the secure Vault store.

    The user_id is injected from the state. Do not ask the user for it.
    Do not add args that are not related to the service.

    For GitHub, you can provide github_org and github_repo.
    For Jira, you can provide a username.

    Example:
        set_user_secrets(user_id='123', service='github', token='ghp_...',
                         github_org='my-org', github_repo='my-repo')
    Example:
        set_user_secrets(user_id='123', service='jira',
                         token='api-token', username='jira-user')

    If an error occurs, the task will be completed and the user will be
    notified of the error.
    """
    if not vault_client:
        return "Error: Vault client is not initialized. Cannot save {service} token."

    if service not in ['github', 'jira']:
        return "Error: service must be 'github' or 'jira'."

    secret_path = f"users/{user_id}/{service.lower()}"
    new_secret_data = {key: value for key, value in details.items() if value}

    try:
        try:
            existing_secret = vault_client.secrets.kv.v2.read_secret_version(
                path=secret_path,
                mount_point=settings.vault.mount_point,
                raise_on_deleted_version=True,
            )
            all_secrets = existing_secret.get("data", {}).get("data", {}) or {}
            all_secrets.update(new_secret_data)
        except hvac.exceptions.InvalidPath:
            # Secret doesn't exist yet, so we'll create it.
            all_secrets = new_secret_data

        vault_client.secrets.kv.v2.create_or_update_secret(
            path=secret_path,
            secret=all_secrets,
            mount_point=settings.vault.mount_point,
        )
        return f"Successfully saved credentials for {service} for user {user_id}."
    except Exception as e:
        logger.error(f"An error occurred while saving {service} token to Vault: {e}")
        return f"An error occurred while saving {service} token to Vault: {e}"