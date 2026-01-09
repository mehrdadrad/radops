"""Handles Role-Based Access Control (RBAC) settings for the application."""
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from config.utils import get_config_path, load_yaml_config

logger = logging.getLogger(__name__)

class UserSettings(BaseModel):
    """Settings for a single user."""
    role: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class OIDCSettings(BaseModel):
    """Settings for generic OIDC integration."""
    enabled: bool = False
    access_token: Optional[str] = None
    userinfo_endpoint: Optional[str] = None
    role_attribute: str = "role"
    cache_ttl_seconds: int = 300


class Auth0Settings(BaseModel):
    """Settings for Auth0 integration."""
    enabled: bool = False
    domain: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    role_attribute: str = "app_metadata.role"
    role_source: str = "metadata"
    cache_ttl_seconds: int = 300


def _validate_config(data: dict) -> dict:
    """Validates and parses the RBAC configuration dictionary."""
    validated = {}

    # Validate OIDC
    oidc_enabled = False
    if "oidc" in data:
        if not isinstance(data["oidc"], dict):
            raise ValueError("'oidc' must be a dictionary")
        oidc_settings = OIDCSettings(**data["oidc"])
        validated["oidc"] = oidc_settings
        oidc_enabled = oidc_settings.enabled

    # Validate Auth0
    auth0_enabled = False
    if "auth0" in data:
        if not isinstance(data["auth0"], dict):
            raise ValueError("'auth0' must be a dictionary")
        auth0_settings = Auth0Settings(**data["auth0"])
        validated["auth0"] = auth0_settings
        auth0_enabled = auth0_settings.enabled

    # Validate users
    if "users" not in data:
        if not oidc_enabled and not auth0_enabled:
            raise ValueError("`users` not found in rbac.yaml")
        validated["users"] = {}
    elif not isinstance(data["users"], dict):
        raise ValueError("'users' must be a dictionary")
    else:
        users = {}
        for k, v in data["users"].items():
            users[k] = UserSettings(**v)
        validated["users"] = users

    # Validate role_permissions
    if "role_permissions" not in data:
        raise ValueError("`role_permissions` not found in rbac.yaml")
    
    if not isinstance(data["role_permissions"], dict):
        raise ValueError("'role_permissions' must be a dictionary")
    
    role_permissions = {}
    for role, perms in data["role_permissions"].items():
        if not isinstance(perms, list):
            raise ValueError(f"Permissions for role '{role}' must be a list")
        role_permissions[role] = [str(p) for p in perms]
    validated["role_permissions"] = role_permissions

    # Validate reload_interval
    if "reload_interval_seconds" in data:
        interval = data["reload_interval_seconds"]
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("'reload_interval_seconds' must be a positive integer")
        validated["reload_interval_seconds"] = interval

    return validated


def yaml_config_settings_source(settings_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    A settings source that loads variables from a YAML file
    at the project's root.
    """
    config_path = get_config_path("rbac.yaml")
    data = load_yaml_config(config_path)
    return _validate_config(data)


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
    users: Dict[str, UserSettings] = Field(default_factory=dict)
    role_permissions: Dict[str, list[str]] = Field(default_factory=dict)
    reload_interval_seconds: int = Field(default=60)
    oidc: OIDCSettings = Field(default_factory=OIDCSettings)
    auth0: Auth0Settings = Field(default_factory=Auth0Settings)

    _watcher_thread: Optional[threading.Thread] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_watcher()

    def get_user(self, user_id: str) -> Optional[UserSettings]:
        """
        Retrieves a user by their ID.

        Args:
            user_id: The ID of the user (typically email).

        Returns:
            The UserSettings object if found, None otherwise.
        """
        return self.users.get(user_id)

    def _start_watcher(self):
        """Starts a background thread to watch for config changes."""
        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()

    def _watch_loop(self):
        """Polls the rbac.yaml file for changes every minute."""
        try:
            config_path = get_config_path("rbac.yaml")
            last_mtime = os.path.getmtime(config_path)
        except OSError as e:
            logger.warning("RBAC watcher failed to initialize: %s", e)
            return

        while True:
            time.sleep(self.reload_interval_seconds)
            try:
                current_mtime = os.path.getmtime(config_path)
                if current_mtime > last_mtime:
                    logger.info("Detected change in rbac.yaml. Reloading...")
                    last_mtime = current_mtime
                    self.reload(config_path)
            except Exception as e:
                logger.error("Error in RBAC watcher loop: %s", e)

    def reload(self, config_path: str = None):
        """Reloads the RBAC settings from the YAML file."""
        try:
            if config_path is None:
                config_path = get_config_path("rbac.yaml")
            new_data = load_yaml_config(config_path)
            
            validated = _validate_config(new_data)

            # Apply changes
            self.users = validated["users"]
            self.role_permissions = validated["role_permissions"]
            if "oidc" in validated:
                self.oidc = validated["oidc"]
            if "auth0" in validated:
                self.auth0 = validated["auth0"]
            if "reload_interval_seconds" in validated:
                self.reload_interval_seconds = validated["reload_interval_seconds"]

            logger.info("RBAC settings reloaded successfully.")
        except (ValidationError, ValueError) as e:
            logger.error("Validation failed for new RBAC settings: %s", e)
        except Exception as e:
            logger.error("Failed to reload RBAC settings: %s", e)

# Instantiate the settings
try:
    rbac_settings = RBACSettings()
except (ValidationError, ValueError) as e:
    print("The application failed to start because of invalid configuration in 'rbac.yaml'.\n", file=sys.stderr)

    if isinstance(e, ValidationError):
        for error in e.errors():
            field_path = " -> ".join(str(x) for x in error['loc'])
            message = error['msg']
            print(f"  • {field_path}: {message}", file=sys.stderr)
    else:
        print(f"  • Error: {e}", file=sys.stderr)

    print("\nPlease verify your 'rbac.yaml' file matches the expected structure.", file=sys.stderr)
    print("For detailed instructions, please refer to 'docs/rbac_guide.md'.", file=sys.stderr)
    sys.exit(1)