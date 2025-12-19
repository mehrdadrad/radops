import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
from console import main


class TestConsole(unittest.IsolatedAsyncioTestCase):
    @patch("console.input")
    @patch("console.print")
    @patch("console.get_checkpointer")
    @patch("console.run_graph", new_callable=AsyncMock)
    @patch("console.astream_graph_updates")
    @patch("console.mem0_manager")
    @patch("console.Telemetry")
    async def test_console_happy_path(
        self, mock_telemetry, mock_mem0, mock_astream, mock_run_graph, mock_get_cp, mock_print, mock_input
    ):
        """Test the main console loop with valid input."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), mock_redis)
        mock_get_cp.return_value = mock_cp_ctx
        
        # Input sequence: username -> message -> quit
        mock_input.side_effect = ["test_user", "hello", "quit"]
        
        # Mock graph stream
        mock_chunk = MagicMock()
        async def async_gen(*args, **kwargs):
            yield mock_chunk
        mock_astream.side_effect = async_gen

        await main()

        # Verify interactions
        mock_input.assert_has_calls([
            call("Please enter your username: "),
            call("User: "),
            call("User: ")
        ])
        mock_astream.assert_called_once()
        mock_chunk.pretty_print.assert_called_once()
        mock_print.assert_called_with("Goodbye!")
        
        # Verify cleanup
        mock_mem0.close.assert_called()
        mock_telemetry.return_value.shutdown.assert_called()
        mock_redis.aclose.assert_called()

    @patch("console.input")
    @patch("console.print")
    @patch("console.get_checkpointer")
    @patch("console.run_graph", new_callable=AsyncMock)
    @patch("console.Telemetry")
    async def test_console_empty_username(
        self, mock_telemetry, mock_run_graph, mock_get_cp, mock_print, mock_input
    ):
        """Test that empty username exits the application."""
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), AsyncMock())
        mock_get_cp.return_value = mock_cp_ctx
        
        mock_input.return_value = "" # Empty username

        await main()

        mock_print.assert_called_with("Username cannot be empty. Exiting.")
        mock_run_graph.assert_called()

    @patch("console.input")
    @patch("console.print")
    @patch("console.get_checkpointer")
    @patch("console.run_graph", new_callable=AsyncMock)
    @patch("console.Telemetry")
    async def test_console_empty_input_loop(
        self, mock_telemetry, mock_run_graph, mock_get_cp, mock_print, mock_input
    ):
        """Test that empty input is ignored and loop continues."""
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), AsyncMock())
        mock_get_cp.return_value = mock_cp_ctx
        
        # username -> empty input -> quit
        mock_input.side_effect = ["user", "   ", "quit"]

        await main()

        # Should continue on empty input and then quit
        mock_print.assert_called_with("Goodbye!")
        self.assertEqual(mock_input.call_count, 3)

    @patch("console.input")
    @patch("console.print")
    @patch("console.get_checkpointer")
    @patch("console.run_graph", new_callable=AsyncMock)
    @patch("console.astream_graph_updates")
    @patch("console.Telemetry")
    async def test_console_exception_handling(
        self, mock_telemetry, mock_astream, mock_run_graph, mock_get_cp, mock_print, mock_input
    ):
        """Test exception handling within the main loop."""
        mock_cp_ctx = AsyncMock()
        mock_cp_ctx.__aenter__.return_value = (MagicMock(), AsyncMock())
        mock_get_cp.return_value = mock_cp_ctx
        
        mock_input.side_effect = ["user", "trigger_error", "quit"]
        
        # Raise exception during streaming
        mock_astream.side_effect = Exception("Graph Error")

        await main()

        mock_print.assert_any_call("An error occurred: Graph Error")
        mock_print.assert_any_call("Goodbye!")