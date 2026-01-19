"""
This module contains tests for the FastAPI server.
"""
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Mock config.server before importing server to avoid SystemExit
# when configuration files are missing in the test environment.
mock_settings = MagicMock()
mock_settings.auth_disabled = True
mock_settings.service_api_key = "test"
mock_settings.service_token = "test"
mock_settings.skip_initial_sync = True
mock_settings.plain_message = False
mock_settings.host = "0.0.0.0"
mock_settings.port = 8005

mock_config = MagicMock()
mock_config.server_settings = mock_settings
sys.modules["config.server"] = mock_config

from fastapi.testclient import TestClient

from server import app


class TestServer(unittest.TestCase):
    """
    Test cases for the FastAPI server.
    """

    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.mem0_manager")
    @patch("server.telemetry")
    @patch("server.ToolRegistry")
    def test_lifespan(
        self, mock_tool_registry, mock_telemetry, mock_mem0, mock_run_graph, mock_get_cp
    ):
        """Test application startup and shutdown logic."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_cp_ctx.__aexit__.return_value = None
        mock_get_cp.return_value = mock_cp_ctx

        mock_run_graph.return_value = MagicMock()
        mock_mem0.close = AsyncMock()
        mock_tool_registry.return_value.close = AsyncMock()

        # TestClient context triggers lifespan
        with TestClient(app):
            # Startup checks
            mock_get_cp.assert_called_once()
            mock_run_graph.assert_called_once()

        # Shutdown checks
        mock_mem0.close.assert_called_once()
        mock_telemetry.shutdown.assert_called_once()
        mock_redis.aclose.assert_called_once()
        mock_tool_registry.return_value.close.assert_called_once()

    @patch("server.server_settings.auth_disabled", True)
    @patch("server.get_user_role")
    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.astream_graph_updates")
    @patch("server.mem0_manager")
    @patch("server.telemetry")
    @patch("server.ToolRegistry")
    def test_websocket_chat(
        self, mock_tool_registry, mock_telemetry, mock_mem0, mock_astream, mock_run_graph, mock_get_cp, mock_get_user_role
    ):
        """Test WebSocket chat flow."""
        # Setup mocks
        mock_get_user_role.return_value = "user"
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_get_cp.return_value = mock_cp_ctx
        mock_run_graph.return_value = MagicMock()
        mock_mem0.close = AsyncMock()
        mock_tool_registry.return_value.close = AsyncMock()

        # Mock streaming response
        mock_msg = MagicMock()
        mock_msg.content = "Hello User"
        mock_msg.tool_calls = []

        async def async_gen(*args, **kwargs):
            yield mock_msg, {}

        mock_astream.side_effect = async_gen

        with TestClient(app) as client:
            with client.websocket_connect("/ws/test_user") as websocket:
                websocket.send_text("Hi")

                # Receive response
                data = websocket.receive_text()
                self.assertEqual(data, "Hello User")

                # Receive end of turn
                eot = websocket.receive_text()
                self.assertEqual(eot, "\x03")

    @patch("server.server_settings.auth_disabled", True)
    @patch("server.server_settings.plain_message", True)
    @patch("server.get_user_role")
    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.astream_graph_updates")
    @patch("server.mem0_manager")
    @patch("server.telemetry")
    @patch("server.ToolRegistry")
    def test_websocket_tool_execution(
        self, mock_tool_registry, mock_telemetry, mock_mem0, mock_astream, mock_run_graph, mock_get_cp, mock_get_user_role
    ):
        """Test WebSocket handling of tool calls."""
        mock_get_user_role.return_value = "user"
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_get_cp.return_value = mock_cp_ctx
        mock_run_graph.return_value = MagicMock()
        mock_mem0.close = AsyncMock()
        mock_tool_registry.return_value.close = AsyncMock()

        # Mock tool call response
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = [{"name": "my_tool"}]

        async def async_gen(*args, **kwargs):
            yield mock_msg, {}

        mock_astream.side_effect = async_gen

        with TestClient(app) as client:
            with client.websocket_connect("/ws/test_user") as websocket:
                websocket.send_text("Run tool")

                data = websocket.receive_text()
                self.assertIn("Running tool: my_tool", data)

                eot = websocket.receive_text()
                self.assertEqual(eot, "\x03")

    @patch("server.server_settings.skip_initial_sync", True)
    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.mem0_manager")
    @patch("server.telemetry")
    @patch("server.ToolRegistry")
    def test_lifespan_skip_sync(
        self, mock_tool_registry, mock_telemetry, mock_mem0, mock_run_graph, mock_get_cp
    ):
        """Test lifespan with SKIP_INITIAL_SYNC env var."""
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_cp_ctx.__aexit__.return_value = None
        mock_get_cp.return_value = mock_cp_ctx
        
        mock_run_graph.return_value = MagicMock()
        mock_mem0.close = AsyncMock()
        mock_tool_registry.return_value.close = AsyncMock()

        with TestClient(app):
            pass
            
        # Verify ToolRegistry called with skip_initial_sync=True
        _, kwargs = mock_tool_registry.call_args
        self.assertTrue(kwargs.get("skip_initial_sync"))