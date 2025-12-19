import unittest
from unittest.mock import MagicMock, patch


from langchain_core.messages import HumanMessage, AIMessage
from core.graph import (
    agent_node,
    route_after_worker,
    route_workflow,
    supervisor_node,
    tools_condition,
)


class TestGraph(unittest.TestCase):
    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")  # Mock telemetry to avoid side effects
    def test_agent_node(self, mock_telemetry, mock_settings, mock_llm_factory):
        """Test that agent_node correctly invokes the LLM with tools."""
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
        }
        tools = [MagicMock()]  # Mock tools list

        # Execute
        result = agent_node(state, tools)

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
        mock_output = MagicMock()
        mock_output.next_worker = "network_specialist"
        mock_structured_llm.invoke.return_value = mock_output

        state = {
            "messages": [HumanMessage(content="Network issue")],
            "user_id": "test_user",
        }

        # Execute
        result = supervisor_node(state)

        # Verify
        self.assertEqual(result["next_worker"], "network_specialist")

    def test_route_workflow(self):
        """Test the routing logic from the supervisor."""
        self.assertEqual(
            route_workflow({"next_worker": "network_specialist"}), "react_agent"
        )
        self.assertEqual(
            route_workflow({"next_worker": "atomic_tool"}), "common_agent"
        )
        self.assertEqual(
            route_workflow({"next_worker": "common_agent"}), "common_agent"
        )
        self.assertEqual(route_workflow({"next_worker": "unknown"}), "end")

    def test_route_after_worker(self):
        """Test routing logic after a worker agent finishes."""
        # Case 1: Tool calls present -> route to tools
        tool_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "test", "args": {}, "id": "test_tool_id"}
            ],
        )
        self.assertEqual(route_after_worker({"messages": [tool_msg]}), "tools")

        # Case 2: Supervisor escalation keyword -> route to supervisor
        escalate_msg = AIMessage(content="I need to escalate to the supervisor.")
        self.assertEqual(
            route_after_worker({"messages": [escalate_msg]}), "supervisor"
        )

        # Case 3: Normal response -> end
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
