"""Authentication and authorization logic for the agent."""
import re

from config.rbac import rbac_settings as settings

def get_user_role(user_id: str) -> str | None:
    """
    Retrieves the role for a given user_id.

    Args:
        user_id: The ID of the user.

    Returns: The user's role, or 'guest' if the user is not found.
    """
    user = settings.get_user(user_id)  # pylint: disable=no-member
    return user.role if user else None 

def is_tool_authorized(tool_name: str, user_id: str) -> bool:
    """
    Checks if a user is authorized to use a specific tool based on their role.
    Supports exact tool names and regex patterns.
    """
    user_role = get_user_role(user_id)
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
