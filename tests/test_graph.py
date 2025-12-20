import unittest
from unittest.mock import AsyncMock, MagicMock, patch


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from core.state import SupervisorAgentOutput
from core.graph import (
    create_agent,
    authorize_tools,
    check_end_status,
    custom_error_handler,
    manage_memory_node,
    route_after_worker,
    route_back_from_tool,
    route_workflow,
    supervisor_node,
    tools_condition,
)


class TestGraph(unittest.IsolatedAsyncioTestCase):
    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")  # Mock telemetry to avoid side effects
    def test_create_agent(self, mock_telemetry, mock_settings, mock_llm_factory):
        """Test that create_agent correctly invokes the LLM with tools."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_factory.return_value = mock_llm

        expected_response = AIMessage(content="Test response")
        mock_llm_with_tools.invoke.return_value = expected_response

        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "test_user",
            "relevant_memories": "User likes python.",
            "response_metadata": {},
        }
        tools = [MagicMock()]  # Mock tools list

        # Execute
        agent_node = create_agent("common", "system prompt", tools)
        result = agent_node(state)

        # Verify
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], expected_response)

        # Check if tools were bound
        mock_llm.bind_tools.assert_called_with(tools)

        # Check if invoke was called
        mock_llm_with_tools.invoke.assert_called()

        # Verify telemetry was updated
        mock_telemetry.update_counter.assert_called_with(
            "agent.invocations.total", attributes={"agent": "common"}
        )

    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    def test_supervisor_node(self, mock_settings, mock_llm_factory):
        """Test that supervisor_node returns the correct next_worker."""
        # Setup
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_factory.return_value = mock_llm

        # Mock the structured output (SupervisorAgentOutput)
        mock_output = MagicMock(
            spec=SupervisorAgentOutput,
            next_worker=MagicMock(value="network_agent"),
        )
        mock_structured_llm.invoke.return_value = mock_output

        state = {
            "messages": [HumanMessage(content="Network issue")],
            "user_id": "test_user",
            "response_metadata": {},
        }

        # Execute
        result = supervisor_node(state)

        # Verify
        self.assertEqual(result["next_worker"], "network_agent")

    def test_route_workflow(self):
        """Test the routing logic from the supervisor."""
        self.assertEqual(
            route_workflow({"next_worker": "network_agent"}), "network_agent"
        )
        self.assertEqual(
            route_workflow({"next_worker": "common_agent"}), "common_agent"
        )

    def test_route_after_worker(self):
        """Test routing logic after a worker agent finishes."""
        # Tool calls present -> route to tools
        tool_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "test", "args": {}, "id": "test_tool_id"}
            ],
        )
        self.assertEqual(route_after_worker({"messages": [tool_msg]}), "tools")

        # Supervisor escalation keyword -> route to supervisor
        escalate_msg = AIMessage(content="I need to escalate to the supervisor.")
        self.assertEqual(
            route_after_worker({"messages": [escalate_msg]}), "supervisor"
        )

        # Normal response -> end
        end_msg = AIMessage(content="Here is the answer.")
        self.assertEqual(route_after_worker({"messages": [end_msg]}), "end")

    def test_tools_condition(self):
        """Test the basic tools condition check."""
        tool_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "test", "args": {}, "id": "test_tool_id"}
            ],
        )
        self.assertEqual(tools_condition({"messages": [tool_msg]}), "tools")

        text_msg = AIMessage(content="text")
        self.assertEqual(tools_condition({"messages": [text_msg]}), "end")

    def test_route_back_from_tool(self):
        """Test routing logic returning from tools."""
        self.assertEqual(route_back_from_tool({"next_worker": "common_agent"}), "common_agent")
        self.assertEqual(route_back_from_tool({"next_worker": "network_agent"}), "network_agent")
        self.assertEqual(route_back_from_tool({"next_worker": "cloud_agent"}), "cloud_agent")

    def test_check_end_status(self):
        """Test check_end_status logic."""
        self.assertEqual(check_end_status({"end_status": "end"}), "end")
        self.assertEqual(check_end_status({"end_status": "continue"}), "continue")

    def test_custom_error_handler(self):
        """Test the custom error handler for tools."""
        error = ValueError("Something went wrong")
        result = custom_error_handler(error)
        self.assertIn("The tool failed to execute", result)
        self.assertIn("Something went wrong", result)

    @patch("core.graph.is_tool_authorized")
    async def test_authorize_tools(self, mock_is_authorized):
        """Test tool authorization logic."""
        # Authorized
        mock_is_authorized.return_value = True
        mock_handler = AsyncMock(return_value="Success")
        request = MagicMock()
        request.tool_call = {"name": "test_tool", "id": "123"}
        request.state = {"user_id": "user1"}

        result = await authorize_tools(request, mock_handler)
        self.assertEqual(result, "Success")
        mock_handler.assert_called_once_with(request)

        # Unauthorized
        mock_is_authorized.return_value = False
        mock_handler.reset_mock()

        result = await authorize_tools(request, mock_handler)
        self.assertIsInstance(result, ToolMessage)
        self.assertIn("unauthorized", result.content)
        self.assertEqual(result.status, "error")
        mock_handler.assert_not_called()

    @patch("core.graph.get_mem0_client")
    @patch("core.graph.settings")
    async def test_manage_memory_node(self, mock_settings, mock_get_mem0):
        """Test memory management node."""
        # Setup mem0 mock
        mock_mem0 = MagicMock()
        mock_mem0.add = AsyncMock()
        mock_mem0.search = AsyncMock(
            return_value={"results": [{"memory": "User likes AI"}]}
        )
        mock_get_mem0.return_value = mock_mem0

        # Setup settings
        mock_settings.mem0.excluded_tools = []
        mock_settings.memory.summarization.keep_message = 2

        state = {
            "messages": [
                HumanMessage(content="Hi", id="1"),
                AIMessage(content="Hello", id="2"),
                HumanMessage(content="Search this", id="3"),
            ],
            "user_id": "user1",
        }

        result = await manage_memory_node(state)

        self.assertIn("relevant_memories", result)
        self.assertIn("User likes AI", result["relevant_memories"])
        mock_mem0.add.assert_called()
        mock_mem0.search.assert_called()
