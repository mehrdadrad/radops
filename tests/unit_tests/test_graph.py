import unittest
from unittest.mock import AsyncMock, MagicMock, patch


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from core.state import SupervisorAgentOutput
from core.graph import (
    create_agent,
    authorize_tools,
    check_end_status,
    custom_error_handler,
    detect_tool_loop,
    manage_memory_node,
    route_after_worker,
    route_back_from_tool,
    route_workflow,
    supervisor_node,
    system_node,
    tools_condition,
    sanitize_tool_calls
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
            response_to_user="Response for user",
            detected_requirements=["Requirement 1", "Requirement 2"],
            completed_steps=["Step 1", "Step 2"],
            is_fully_completed=True,
            instructions_for_worker="Instructions for worker",
            original_request="Original user request",
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

    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")
    def test_system_node(self, mock_telemetry, mock_settings, mock_llm_factory):
        """Test that system_node correctly invokes the LLM with tools."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_factory.return_value = mock_llm

        expected_response = AIMessage(content="System response")
        mock_llm_with_tools.invoke.return_value = expected_response

        # Mock settings
        mock_settings.agent.system.llm_profile = "test-profile"

        state = {
            "messages": [HumanMessage(content="Clear memory")],
            "user_id": "test_user",
            "relevant_memories": "",
            "response_metadata": {},
        }
        tools = [MagicMock()]

        # Execute
        result = system_node(state, tools)

        # Verify
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], expected_response)
        self.assertIn("system", result["response_metadata"]["nodes"])

        # Check if tools were bound
        mock_llm.bind_tools.assert_called_with(tools)

        # Check if invoke was called
        mock_llm_with_tools.invoke.assert_called()

        # Verify telemetry
        mock_telemetry.update_counter.assert_called_with(
            "agent.invocations.total", attributes={"agent": "system"}
        )

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

        # Normal response -> end
        end_msg = AIMessage(content="Here is the answer.")
        self.assertEqual(route_after_worker({"messages": [end_msg]}), "supervisor")

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

    def test_detect_tool_loop(self):
        """Test the tool loop detection logic."""
        # Case 1: No loop (not enough history)
        state_short = {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "tool_a", "args": {"x": 1}, "id": "1"}])
            ]
        }
        self.assertFalse(detect_tool_loop(state_short))

        # Case 2: Identical loop
        # limit is 3 by default. We need 3 identical calls.
        tool_call = {"name": "tool_a", "args": {"x": 1}, "id": "1"}
        state_identical = {
            "messages": [
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[tool_call]),
            ]
        }
        self.assertTrue(detect_tool_loop(state_identical))

        # Case 3: Alternating loop (A, B, A, B)
        call_a = {"name": "tool_a", "args": {}, "id": "1"}
        call_b = {"name": "tool_b", "args": {}, "id": "2"}
        state_alternating = {
            "messages": [
                AIMessage(content="", tool_calls=[call_a]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[call_b]),
                ToolMessage(content="res", tool_call_id="2", name="tool_b"),
                AIMessage(content="", tool_calls=[call_a]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[call_b]),
            ]
        }
        self.assertTrue(detect_tool_loop(state_alternating))

    def test_route_after_worker_loop_detection(self):
        """Test that route_after_worker catches loops."""
        tool_call = {"name": "tool_a", "args": {"x": 1}, "id": "1"}
        # Create a state that triggers the identical loop check (limit=3)
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[tool_call]),
            ]
        }
        # Should return supervisor instead of tools because of loop
        self.assertEqual(route_after_worker(state), "supervisor")

    def test_route_back_from_tool(self):
        """Test routing logic returning from tools."""
        self.assertEqual(route_back_from_tool({"next_worker": "common_agent"}), "common_agent")
        self.assertEqual(route_back_from_tool({"next_worker": "network_agent"}), "network_agent")
        self.assertEqual(route_back_from_tool({"next_worker": "cloud_agent"}), "cloud_agent")

        # Test escalation
        escalation_state = {
            "messages": [
                ToolMessage(content="Escalating", tool_call_id="1", name="system__submit_work")
            ],
            "next_worker": "network_agent"
        }
        self.assertEqual(route_back_from_tool(escalation_state), "supervisor")

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

class TestSanitizeToolCalls(unittest.TestCase):
    def test_basic_conversation(self):
        """Test that a basic conversation without tool calls is preserved."""
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        sanitized = sanitize_tool_calls(messages)
        self.assertEqual(sanitized, messages)

    def test_valid_tool_call(self):
        """Test that a valid tool call followed by a matching tool message is preserved."""
        messages = [
            HumanMessage(content="do task"),
            AIMessage(content="", tool_calls=[{'id': 'call_1', 'name': 'tool1', 'args': {}}], id='ai_1'),
            ToolMessage(content="result", tool_call_id='call_1', name='tool1'),
            AIMessage(content="done"),
        ]
        sanitized = sanitize_tool_calls(messages)
        self.assertEqual(sanitized, messages)

    def test_invalid_tool_calls_attribute(self):
        """Test that AIMessage with invalid_tool_calls is sanitized to text-only."""
        msg = AIMessage(content="", id='ai_1')
        # Simulate invalid tool calls (e.g. parsing error)
        msg.invalid_tool_calls = [{'type': 'invalid_tool_call', 'id': 'call_1', 'name': 'tool1', 'args': 'bad_json', 'error': 'error'}]
        
        messages = [
            HumanMessage(content="do task"),
            msg
        ]
        
        sanitized = sanitize_tool_calls(messages)
        
        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[1], AIMessage)
        self.assertEqual(sanitized[1].content, "...")
        self.assertFalse(sanitized[1].tool_calls)
        # Ensure invalid_tool_calls attribute is not present or empty in the new message
        self.assertFalse(getattr(sanitized[1], 'invalid_tool_calls', []))

    def test_missing_tool_message(self):
        """Test that AIMessage with tool_calls but no following ToolMessage is sanitized."""
        messages = [
            HumanMessage(content="do task"),
            AIMessage(content="thought", tool_calls=[{'id': 'call_1', 'name': 'tool1', 'args': {}}], id='ai_1'),
        ]
        
        sanitized = sanitize_tool_calls(messages)
        
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[1].content, "thought")
        self.assertEqual(sanitized[1].tool_calls, [])

    def test_partial_tool_message_missing(self):
        """Test that if not all tool calls have responses, the AI message is sanitized and partial responses removed."""
        messages = [
            HumanMessage(content="do task"),
            AIMessage(content="", tool_calls=[
                {'id': 'call_1', 'name': 'tool1', 'args': {}},
                {'id': 'call_2', 'name': 'tool2', 'args': {}}
            ], id='ai_1'),
            ToolMessage(content="result1", tool_call_id='call_1', name='tool1'),
        ]
        
        sanitized = sanitize_tool_calls(messages)
        
        # Should strip tool calls from AIMessage and remove the partial ToolMessage
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[1].content, "...")
        self.assertEqual(sanitized[1].tool_calls, [])

    def test_orphaned_tool_message(self):
        """Test that a ToolMessage without a preceding AIMessage is removed."""
        messages = [
            HumanMessage(content="hi"),
            ToolMessage(content="result", tool_call_id='call_1', name='tool1'),
        ]
        
        sanitized = sanitize_tool_calls(messages)
        
        self.assertEqual(len(sanitized), 1)
        self.assertIsInstance(sanitized[0], HumanMessage)

    def test_interleaved_valid_and_invalid(self):
        """Test a mix of valid pairs, invalid pairs, and orphaned messages."""
        messages = [
            HumanMessage(content="start"),
            # Valid pair
            AIMessage(content="", tool_calls=[{'id': 'call_1', 'name': 't1', 'args': {}}], id='ai_1'),
            ToolMessage(content="r1", tool_call_id='call_1', name='t1'),
            # Invalid (missing tool msg)
            AIMessage(content="thinking", tool_calls=[{'id': 'call_2', 'name': 't2', 'args': {}}], id='ai_2'),
            # Orphaned tool msg
            ToolMessage(content="r3", tool_call_id='call_3', name='t3'),
            AIMessage(content="end"),
        ]
        
        sanitized = sanitize_tool_calls(messages)
        
        self.assertEqual(len(sanitized), 5)
        # Human
        self.assertIsInstance(sanitized[0], HumanMessage)
        # AI (valid)
        self.assertEqual(sanitized[1].id, 'ai_1')
        self.assertTrue(sanitized[1].tool_calls)
        # Tool (valid)
        self.assertIsInstance(sanitized[2], ToolMessage)
        self.assertEqual(sanitized[2].tool_call_id, 'call_1')
        # AI (sanitized, originally ai_2)
        self.assertEqual(sanitized[3].id, 'ai_2')
        self.assertFalse(sanitized[3].tool_calls)
        self.assertEqual(sanitized[3].content, "thinking")
        # AI (end) - The orphaned tool message 'r3' is skipped.
        self.assertEqual(sanitized[4].content, "end")