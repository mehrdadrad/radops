import unittest
from unittest.mock import MagicMock, patch, call
import logging
import hvac

from src.utils.status_generator import StatusGenerator
from src.utils.logger import _parse_size, initialize_logger
from src.utils.vault_resolver import resolve_vault_secrets
from src.utils import secrets

class TestStatusGenerator(unittest.TestCase):
    def test_icons_and_verbs(self):
        # Test AWS List
        res = StatusGenerator.parse_tool_call("aws_list_ec2", {})
        self.assertIn("☁️", res)
        self.assertIn("Listing", res)
        self.assertIn("Ec2", res)

    def test_network_ping(self):
        res = StatusGenerator.parse_tool_call("network_ping", {"target": "8.8.8.8"})
        self.assertIn("🌐", res)
        self.assertIn("Pinging", res)
        self.assertIn("8.8.8.8", res)

    def test_github_create(self):
        res = StatusGenerator.parse_tool_call("github_create_issue", {"title": "Bug"})
        self.assertIn("🐙", res)
        self.assertIn("Creating", res)
        self.assertIn("Issue", res)
        self.assertIn("Bug", res)

    def test_unknown_tool(self):
        res = StatusGenerator.parse_tool_call("unknown_tool_action", {})
        self.assertIn("⚙️", res) # Default icon
        self.assertIn("Processing", res) # Default verb

    def test_context_formatting(self):
        # String context
        res = StatusGenerator.parse_tool_call("test", {"arg": "short_val"})
        self.assertIn(": short_val", res)
        
        # List context
        res = StatusGenerator.parse_tool_call("test", {"items": [1, 2, 3]})
        self.assertIn(": 3 items", res)

class TestLoggerUtils(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(_parse_size("10B"), 10)
        self.assertEqual(_parse_size("1 KB"), 1024)
        self.assertEqual(_parse_size("1MB"), 1024**2)
        self.assertEqual(_parse_size("2 GB"), 2 * 1024**3)
        self.assertEqual(_parse_size("invalid"), 10 * 1024**2) # Default

    @patch("src.utils.logger.logging")
    @patch("src.utils.logger.RotatingFileHandler")
    @patch("src.utils.logger.settings")
    def test_initialize_logger_file(self, mock_settings, mock_handler, mock_logging):
        mock_settings.logging.level = "DEBUG"
        mock_settings.logging.file = "/tmp/test.log"
        mock_settings.logging.rotation = "1 MB"

        initialize_logger()

        mock_handler.assert_called_with("/tmp/test.log", maxBytes=1024**2, backupCount=5)
        mock_logging.basicConfig.assert_called()
        # Check level setting
        _, kwargs = mock_logging.basicConfig.call_args
        self.assertEqual(kwargs["level"], "DEBUG")

    @patch("src.utils.logger.logging")
    @patch("src.utils.logger.settings")
    def test_initialize_logger_stream(self, mock_settings, mock_logging):
        mock_settings.logging.level = "INFO"
        mock_settings.logging.file = None

        initialize_logger()

        # Should use StreamHandler
        mock_logging.StreamHandler.assert_called()
        mock_logging.basicConfig.assert_called()

class TestVaultResolver(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mount_point = "secret"

    def test_resolve_simple_string(self):
        val = "simple"
        res = resolve_vault_secrets(val, self.mock_client, self.mount_point)
        self.assertEqual(res, "simple")

    def test_resolve_full_match(self):
        # vault:path/to/secret#key
        val = "vault:users/test#token"
        
        # Mock response
        mock_response = {
            "data": {
                "data": {"token": "secret_value"}
            }
        }
        self.mock_client.secrets.kv.v2.read_secret_version.return_value = mock_response

        res = resolve_vault_secrets(val, self.mock_client, self.mount_point)
        self.assertEqual(res, "secret_value")
        self.mock_client.secrets.kv.v2.read_secret_version.assert_called_with(
            path="users/test", mount_point="secret", raise_on_deleted_version=True
        )

    def test_resolve_full_match_whole_secret(self):
        # vault:path/to/secret (no key)
        val = "vault:users/test"
        
        mock_response = {
            "data": {
                "data": {"key1": "val1"}
            }
        }
        self.mock_client.secrets.kv.v2.read_secret_version.return_value = mock_response

        res = resolve_vault_secrets(val, self.mock_client, self.mount_point)
        self.assertEqual(res, "val1") # Returns single value if only one

        # Multiple values
        mock_response["data"]["data"] = {"k1": "v1", "k2": "v2"}
        res = resolve_vault_secrets(val, self.mock_client, self.mount_point)
        self.assertEqual(res, {"k1": "v1", "k2": "v2"})

    def test_resolve_partial_interpolation(self):
        val = "Token is vault:users/test#token end"
        
        mock_response = {
            "data": {
                "data": {"token": "123"}
            }
        }
        self.mock_client.secrets.kv.v2.read_secret_version.return_value = mock_response

        res = resolve_vault_secrets(val, self.mock_client, self.mount_point)
        self.assertEqual(res, "Token is 123 end")

    def test_resolve_recursive(self):
        config = {
            "a": "vault:path/a#k",
            "b": ["vault:path/b#k"],
            "c": "plain"
        }
        
        def side_effect(path, mount_point, raise_on_deleted_version):
            if path == "path/a":
                return {"data": {"data": {"k": "val_a"}}}
            if path == "path/b":
                return {"data": {"data": {"k": "val_b"}}}
            return None

        self.mock_client.secrets.kv.v2.read_secret_version.side_effect = side_effect

        res = resolve_vault_secrets(config, self.mock_client, self.mount_point)
        self.assertEqual(res["a"], "val_a")
        self.assertEqual(res["b"][0], "val_b")
        self.assertEqual(res["c"], "plain")

class TestSecretsModule(unittest.TestCase):
    def setUp(self):
        # Patch the module-level vault_client in src.utils.secrets
        self.patcher = patch("src.utils.secrets.vault_client")
        self.mock_vault_client = self.patcher.start()
        
        self.settings_patcher = patch("src.utils.secrets.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.vault.mount_point = "secret"

    def tearDown(self):
        self.patcher.stop()
        self.settings_patcher.stop()

    def test_get_user_secrets_success(self):
        mock_response = {
            "data": {
                "data": {"api_key": "12345"}
            }
        }
        self.mock_vault_client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        res = secrets.get_user_secrets("user1", "github")
        self.assertEqual(res, {"api_key": "12345"})
        
        self.mock_vault_client.secrets.kv.v2.read_secret_version.assert_called_with(
            path="users/user1/github",
            mount_point="secret",
            raise_on_deleted_version=True
        )

    def test_get_user_secrets_no_client(self):
        # Simulate client not initialized
        with patch("src.utils.secrets.vault_client", None):
            res = secrets.get_user_secrets("user1", "github")
            self.assertIn("error", res)
            self.assertIn("not initialized", res["error"])

            with self.assertRaises(Exception):
                secrets.get_user_secrets("user1", "github", raise_on_error=True)

    def test_get_user_secrets_invalid_path(self):
        self.mock_vault_client.secrets.kv.v2.read_secret_version.side_effect = hvac.exceptions.InvalidPath("missing")
        
        res = secrets.get_user_secrets("user1", "github")
        self.assertIn("error", res)
        self.assertIn("No secrets found", res["error"])

    def test_get_user_secrets_generic_error(self):
        self.mock_vault_client.secrets.kv.v2.read_secret_version.side_effect = Exception("Boom")
        
        res = secrets.get_user_secrets("user1", "github")
        self.assertIn("error", res)
        self.assertIn("An error occurred", res["error"])