"""Authentication and authorization logic for the agent."""
import logging
import re
import time

import httpx

from config.rbac import rbac_settings as settings, UserSettings

logger = logging.getLogger(__name__)

_oidc_cache = {}

_auth0_cache = {}
_auth0_token = {"access_token": None, "expires_at": 0}

async def _get_oidc_user(user_id: str) -> UserSettings | None:
    """Fetches user details from a generic OIDC provider."""
    if not settings.oidc.enabled:
        return None

    now = time.time()
    if user_id in _oidc_cache:
        user, timestamp = _oidc_cache[user_id]
        if now - timestamp < settings.oidc.cache_ttl_seconds:
            return user

    if not settings.oidc.userinfo_endpoint:
        logger.error("OIDC is enabled but userinfo_endpoint is missing.")
        return None

    headers = {
        "Authorization": f"Bearer {settings.oidc.access_token}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(settings.oidc.userinfo_endpoint, headers=headers)
            if resp.status_code == 200:
                data = resp.json()

                role = data
                for key in settings.oidc.role_attribute.split("."):
                    if isinstance(role, dict):
                        role = role.get(key)
                    else:
                        role = None
                        break

                if isinstance(role, str):
                    user = UserSettings(
                        role=role,
                        first_name=data.get("given_name"),
                        last_name=data.get("family_name"),
                    )
                    _oidc_cache[user_id] = (user, now)
                    return user

                logger.warning(
                    "User found in OIDC but no role attribute '%s' found.",
                    settings.oidc.role_attribute
                )
            elif resp.status_code == 401:
                logger.debug("OIDC token invalid or expired.")
            else:
                logger.error(
                    "Error fetching user from OIDC: %s %s",
                    resp.status_code,
                    resp.text
                )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Exception fetching user from OIDC: %s", e)

    return None


async def _get_auth0_management_token() -> str | None:
    """Fetches or refreshes the Auth0 Management API token."""
    if not settings.auth0.enabled:
        return None

    if (
        not settings.auth0.domain
        or not settings.auth0.client_id
        or not settings.auth0.client_secret
    ):
        logger.error(
            "Auth0 is enabled but domain, client_id, or client_secret is missing."
        )
        return None

    now = time.time()
    if _auth0_token["access_token"] and _auth0_token["expires_at"] > now + 60:
        return _auth0_token["access_token"]

    url = f"https://{settings.auth0.domain}/oauth/token"
    payload = {
        "client_id": settings.auth0.client_id,
        "client_secret": settings.auth0.client_secret,
        "audience": f"https://{settings.auth0.domain}/api/v2/",
        "grant_type": "client_credentials",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                _auth0_token["access_token"] = data["access_token"]
                _auth0_token["expires_at"] = now + data["expires_in"]
                return data["access_token"]

            logger.error("Failed to get Auth0 token: %s %s", resp.status_code, resp.text)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Exception getting Auth0 token: %s", e)
    return None


async def _resolve_auth0_role(user_data: dict, client: httpx.AsyncClient, headers: dict) -> str | None:
    """Resolves the user role based on configuration."""
    if settings.auth0.role_source == "metadata":
        # Extract role from nested dictionary path (e.g. app_metadata.role)
        role_val = user_data
        for key in settings.auth0.role_attribute.split("."):
            if isinstance(role_val, dict):
                role_val = role_val.get(key)
            else:
                return None
        return role_val if isinstance(role_val, str) else None

    if settings.auth0.role_source == "native":
        try:
            uid = user_data.get("user_id")
            roles_url = (
                f"https://{settings.auth0.domain}/api/v2/users/{uid}/roles"
            )
            resp_roles = await client.get(roles_url, headers=headers)
            if resp_roles.status_code == 200:
                roles_data = resp_roles.json()
                if roles_data:
                    return roles_data[0].get("name")
        except Exception as role_ex:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to fetch native roles for %s: %s",
                user_data.get("user_id"),
                role_ex
            )
    return None


async def _get_auth0_user(user_id: str) -> UserSettings | None:
    """Fetches user details from Auth0."""
    if not settings.auth0.enabled:
        return None

    now = time.time()
    if user_id in _auth0_cache:
        user, timestamp = _auth0_cache[user_id]
        if now - timestamp < settings.auth0.cache_ttl_seconds:
            return user

    token = await _get_auth0_management_token()
    if not token:
        return None

    url = f"https://{settings.auth0.domain}/api/v2/users-by-email"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"email": user_id}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                logger.error("Error fetching user from Auth0: %s %s", resp.status_code, resp.text)
                return None

            users = resp.json()
            if not users:
                logger.debug("User %s not found in Auth0.", user_id)
                return None

            user_data = users[0]
            role_val = await _resolve_auth0_role(user_data, client, headers)

            if role_val:
                user = UserSettings(
                    role=role_val,
                    first_name=user_data.get("given_name"),
                    last_name=user_data.get("family_name"),
                )
                _auth0_cache[user_id] = (user, now)
                return user

            logger.warning(
                "User %s found in Auth0 but role could not be determined "
                "using source '%s'.",
                user_id,
                settings.auth0.role_source,
            )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Exception fetching user from Auth0: %s", e)

    return None


async def get_user(user_id: str) -> UserSettings | None:
    """
    Retrieves a user by their ID, checking local config, then OIDC, then Auth0.
    """
    user = settings.get_user(user_id)  # pylint: disable=no-member
    if user:
        return user

    user = await _get_oidc_user(user_id)
    if user:
        return user

    return await _get_auth0_user(user_id)


async def get_user_role(user_id: str) -> str | None:
    """
    Retrieves the role for a given user_id.

    Args:
        user_id: The ID of the user.

    Returns: The user's role, or None if the user is not found.
    """
    user = await get_user(user_id)
    return user.role if user else None

async def is_tool_authorized(tool_name: str, user_id: str) -> bool:
    """
    Checks if a user is authorized to use a specific tool based on their role.
    Supports exact tool names and regex patterns.
    """
    user_role = await get_user_role(user_id)
    if not user_role:
        return False

    authorized_tool_names = settings.role_permissions.get(user_role, [])  # pylint: disable=no-member

    # Fast check for exact match
    if tool_name in authorized_tool_names:
        return True

    # Fallback to regex matching for patterns
    for pattern in authorized_tool_names:
        try:
            if re.fullmatch(pattern, tool_name):
                return True
        except re.error:
            # Ignore invalid regex patterns in the config
            continue

    return False
