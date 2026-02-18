import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import time
from core import auth

class TestAuthExtended(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Clear caches before each test
        auth._oidc_cache.clear()
        auth._auth0_cache.clear()
        auth._auth0_token = {"access_token": None, "expires_at": 0}

    @patch("core.auth.settings")
    @patch("core.auth.httpx.AsyncClient")
    async def test_oidc_cache_hit(self, mock_client_cls, mock_settings):
        mock_settings.oidc.enabled = True
        mock_settings.oidc.cache_ttl_seconds = 60
        mock_settings.oidc.userinfo_endpoint = "http://oidc"
        mock_settings.oidc.role_attribute = "role"
        
        # Mock network call
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sub": "u1", "role": "admin"}
        mock_client.get.return_value = mock_response

        # First call - should hit network
        user1 = await auth._get_oidc_user("u1")
        self.assertEqual(user1.role, "admin")
        self.assertEqual(mock_client.get.call_count, 1)

        # Second call - should hit cache
        user2 = await auth._get_oidc_user("u1")
        self.assertEqual(user2.role, "admin")
        self.assertEqual(mock_client.get.call_count, 1)

    @patch("core.auth.settings")
    @patch("core.auth.httpx.AsyncClient")
    async def test_oidc_cache_expiry(self, mock_client_cls, mock_settings):
        mock_settings.oidc.enabled = True
        mock_settings.oidc.cache_ttl_seconds = 0.1 # Short TTL
        mock_settings.oidc.userinfo_endpoint = "http://oidc"
        mock_settings.oidc.role_attribute = "role"
        
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sub": "u1", "role": "admin"}
        mock_client.get.return_value = mock_response

        # First call
        await auth._get_oidc_user("u1")
        self.assertEqual(mock_client.get.call_count, 1)

        # Wait for expiry
        time.sleep(0.2)

        # Second call - should hit network again
        await auth._get_oidc_user("u1")
        self.assertEqual(mock_client.get.call_count, 2)

    @patch("core.auth.settings")
    @patch("core.auth.httpx.AsyncClient")
    async def test_auth0_token_cache(self, mock_client_cls, mock_settings):
        mock_settings.auth0.enabled = True
        mock_settings.auth0.domain = "d"
        mock_settings.auth0.client_id = "c"
        mock_settings.auth0.client_secret = "s"
        
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        
        # Mock token response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # expires_in 100 seconds
        mock_response.json.return_value = {"access_token": "token1", "expires_in": 100}
        mock_client.post.return_value = mock_response

        # First call
        token1 = await auth._get_auth0_management_token()
        self.assertEqual(token1, "token1")
        self.assertEqual(mock_client.post.call_count, 1)

        # Second call - should use cached token
        token2 = await auth._get_auth0_management_token()
        self.assertEqual(token2, "token1")
        self.assertEqual(mock_client.post.call_count, 1)