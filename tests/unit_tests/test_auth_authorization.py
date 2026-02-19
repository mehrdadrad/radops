import unittest
from unittest.mock import patch, AsyncMock
from core import auth

class TestAuthAuthorization(unittest.IsolatedAsyncioTestCase):
    @patch("core.auth.settings")
    @patch("core.auth.get_user_role")
    async def test_is_tool_authorized_exact_match(self, mock_get_user_role, mock_settings):
        """Test authorization with exact tool name matches."""
        mock_get_user_role.return_value = "admin"
        mock_settings.role_permissions = {"admin": ["tool_a", "tool_b"]}
        
        self.assertTrue(await auth.is_tool_authorized("tool_a", "user1"))
        self.assertFalse(await auth.is_tool_authorized("tool_c", "user1"))

    @patch("core.auth.settings")
    @patch("core.auth.get_user_role")
    async def test_is_tool_authorized_regex_match(self, mock_get_user_role, mock_settings):
        """Test authorization with regex patterns."""
        mock_get_user_role.return_value = "operator"
        mock_settings.role_permissions = {"operator": ["system__.*", "network_.*"]}
        
        self.assertTrue(await auth.is_tool_authorized("system__list_tools", "user1"))
        self.assertTrue(await auth.is_tool_authorized("network_ping", "user1"))
        self.assertFalse(await auth.is_tool_authorized("auth_login", "user1"))

    @patch("core.auth.settings")
    @patch("core.auth.get_user_role")
    async def test_is_tool_authorized_no_role(self, mock_get_user_role, mock_settings):
        """Test authorization when user has no role."""
        mock_get_user_role.return_value = None
        self.assertFalse(await auth.is_tool_authorized("tool_a", "user1"))

    @patch("core.auth.settings")
    @patch("core.auth.get_user_role")
    async def test_is_tool_authorized_invalid_regex(self, mock_get_user_role, mock_settings):
        """Test that invalid regex in config does not crash the application."""
        mock_get_user_role.return_value = "user"
        # Invalid regex (unbalanced bracket) shouldn't crash the app
        mock_settings.role_permissions = {"user": ["[invalid_regex", "valid_tool"]}
        
        self.assertTrue(await auth.is_tool_authorized("valid_tool", "user1"))
        self.assertFalse(await auth.is_tool_authorized("anything", "user1"))

    @patch("core.auth.settings")
    @patch("core.auth._get_oidc_user")
    @patch("core.auth._get_auth0_user")
    async def test_get_user_priority(self, mock_auth0, mock_oidc, mock_settings):
        """Test the priority order of user resolution (Local -> OIDC -> Auth0)."""
        # 1. Local user found
        mock_settings.get_user.return_value = "local_user"
        user = await auth.get_user("u1")
        self.assertEqual(user, "local_user")
        mock_oidc.assert_not_called()

        # 2. OIDC user found (Local missing)
        mock_settings.get_user.return_value = None
        mock_oidc.return_value = "oidc_user"
        user = await auth.get_user("u1")
        self.assertEqual(user, "oidc_user")
        mock_auth0.assert_not_called()

        # 3. Auth0 user found (Local & OIDC missing)
        mock_oidc.return_value = None
        mock_auth0.return_value = "auth0_user"
        user = await auth.get_user("u1")
        self.assertEqual(user, "auth0_user")