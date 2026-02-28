import os
import unittest
from unittest.mock import MagicMock, patch
from config.integrations import IntegrationSettings
from config.config import (
    GuardrailsSettings,
    Settings,
    yaml_config_settings_source,
)
from pydantic import ValidationError


class TestIntegrationSettings(unittest.TestCase):
    def setUp(self):
        # Clear environment variables that might interfere
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_from_yaml(self, mock_yaml_load, mock_open):
        """Test loading settings from a YAML file."""
        mock_yaml_load.return_value = {
            "slack": {
                "bot_token": "yaml-bot-token",
                "app_token": "yaml-app-token",
                "log_level": "DEBUG"
            }
        }

        settings = IntegrationSettings()

        self.assertEqual(settings.slack.bot_token, "yaml-bot-token")
        self.assertEqual(settings.slack.app_token, "yaml-app-token")
        self.assertEqual(settings.slack.log_level, "DEBUG")

    def test_env_var_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            "SLACK__BOT_TOKEN": "env-bot-token",
            "SLACK__LOG_LEVEL": "ERROR"
        }):
            # Mock open to raise FileNotFoundError so it skips YAML loading
            with patch("builtins.open", side_effect=FileNotFoundError):
                settings = IntegrationSettings()

                self.assertEqual(settings.slack.bot_token, "env-bot-token")
                self.assertEqual(settings.slack.log_level, "ERROR")

    @patch("config.utils.hvac.Client")
    @patch("config.utils.resolve_vault_secrets")
    @patch("config.integrations.app_settings")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_vault_resolution(
        self, mock_yaml, mock_open, mock_app_settings, mock_resolve, mock_hvac
    ):
        """Test Vault secret resolution logic."""
        # Setup YAML data
        yaml_data = {"slack": {"bot_token": "vault:secret#token"}}
        mock_yaml.return_value = yaml_data

        # Setup Vault config via app_settings mock
        mock_app_settings.vault.url = "http://vault:8200"
        mock_app_settings.vault.token = "root"
        mock_app_settings.vault.mount_point = "secret"

        # Setup HVAC client
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.return_value = mock_client

        # Setup resolution result
        resolved_data = {"slack": {"bot_token": "resolved-token"}}
        mock_resolve.return_value = resolved_data

        settings = IntegrationSettings()

        # Verify resolve_vault_secrets was called
        mock_resolve.assert_called_once()

        # Verify settings contain resolved value
        self.assertEqual(settings.slack.bot_token, "resolved-token")

    @patch("config.utils.hvac.Client")
    @patch("config.integrations.app_settings")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_vault_auth_failure(self, mock_yaml, mock_open, mock_app_settings, mock_hvac):
        """Test behavior when Vault authentication fails."""
        mock_yaml.return_value = {"slack": {"bot_token": "vault:secret#token"}}
        mock_app_settings.vault.url = "http://vault:8200"
        mock_app_settings.vault.token = "root"
        mock_hvac.return_value.is_authenticated.return_value = False

        settings = IntegrationSettings()

        # Should fall back to raw YAML value since resolution was skipped
        self.assertEqual(settings.slack.bot_token, "vault:secret#token")


class TestMainConfig(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_guardrails_validation(self):
        """Test GuardrailsSettings validation logic."""
        # Valid: disabled
        settings = GuardrailsSettings(enabled=False)
        self.assertFalse(settings.enabled)

        # Valid: enabled with required fields
        settings = GuardrailsSettings(
            enabled=True, llm_profile="test", prompt_file="test.txt"
        )
        self.assertTrue(settings.enabled)

        # Invalid: enabled missing llm_profile
        with self.assertRaises(ValidationError):
            GuardrailsSettings(enabled=True, prompt_file="test.txt")

        # Invalid: enabled missing prompt_file
        with self.assertRaises(ValidationError):
            GuardrailsSettings(enabled=True, llm_profile="test")

    @patch("config.config.process_vault_secrets")
    @patch("config.config.load_yaml_config")
    @patch("config.config.get_config_path")
    def test_yaml_source(self, mock_get_path, mock_load_yaml, mock_process_vault):
        """Test the YAML settings source loader."""
        mock_get_path.return_value = "/tmp/config.yaml"
        mock_load_yaml.return_value = {"vault": {"url": "http://vault"}}
        mock_process_vault.return_value = {"processed": True}

        # Call the source function
        result = yaml_config_settings_source(Settings)

        mock_get_path.assert_called_with("config.yaml")
        mock_load_yaml.assert_called_with("/tmp/config.yaml")
        mock_process_vault.assert_called()
        self.assertEqual(result, {"processed": True})

    @patch("config.config.yaml_config_settings_source")
    def test_settings_load(self, mock_yaml_source):
        """Test loading the main Settings object."""
        # Mock the YAML source to return minimal valid data
        mock_yaml_source.return_value = {
            "llm": {
                "default_profile": "test",
                "profiles": {"test": {"provider": "openai", "model": "gpt-4"}},
            },
            "discovery": {
                "embedding_profile": "test_embedding",
            },
        }

        settings = Settings()
        self.assertEqual(settings.llm.default_profile, "test")
        self.assertEqual(settings.llm.profiles["test"].provider, "openai")