"""
This module contains tests for the FastAPI server.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from server import app


class TestServer(unittest.TestCase):
    """
    Test cases for the FastAPI server.
    """

    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.mem0_manager")
    @patch("server.Telemetry")
    def test_lifespan(
        self, mock_telemetry, mock_mem0, mock_run_graph, mock_get_cp
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

        # TestClient context triggers lifespan
        with TestClient(app):
            # Startup checks
            mock_get_cp.assert_called_once()
            mock_run_graph.assert_called_once()

        # Shutdown checks
        mock_mem0.close.assert_called_once()
        mock_telemetry.return_value.shutdown.assert_called_once()
        mock_redis.aclose.assert_called_once()

    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.astream_graph_updates")
    @patch("server.mem0_manager")
    @patch("server.Telemetry")
    def test_websocket_chat(
        self, mock_telemetry, mock_mem0, mock_astream, mock_run_graph, mock_get_cp
    ):
        """Test WebSocket chat flow."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_get_cp.return_value = mock_cp_ctx
        mock_run_graph.return_value = MagicMock()
        mock_mem0.close = AsyncMock()

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

    @patch("server.get_checkpointer")
    @patch("server.run_graph", new_callable=AsyncMock)
    @patch("server.astream_graph_updates")
    @patch("server.mem0_manager")
    @patch("server.Telemetry")
    def test_websocket_tool_execution(
        self, mock_telemetry, mock_mem0, mock_astream, mock_run_graph, mock_get_cp
    ):
        """Test WebSocket handling of tool calls."""
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_get_cp.return_value = mock_cp_ctx
        mock_run_graph.return_value = MagicMock()
        mock_mem0.close = AsyncMock()

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