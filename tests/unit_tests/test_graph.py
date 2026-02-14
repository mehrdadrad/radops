import unittest
from unittest.mock import AsyncMock, MagicMock, patch


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage
from core.state import SupervisorAgentPlanOutput
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
    sanitize_tool_calls,
    enforce_plan,
    get_detected_requirements,
    update_step_status,
    check_completion,
    filter_tools,
    contains_sensitive_data,
    delete_tool_messages,
    auditor_node
)


class TestGraph(unittest.IsolatedAsyncioTestCase):
    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")  # Mock telemetry to avoid side effects
    async def test_create_agent(self, mock_telemetry, mock_settings, mock_llm_factory):
        """Test that create_agent correctly invokes the LLM with tools."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_factory.return_value = mock_llm

        expected_response = AIMessage(content="Test response")
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=expected_response)

        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "test_user",
            "relevant_memories": "User likes python.",
            "response_metadata": {},
        }
        tools = [MagicMock()]  # Mock tools list

        # Execute
        agent_node = create_agent("common", "system prompt", tools)
        result = await agent_node(state)

        # Verify
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], expected_response)

        # Check if tools were bound
        mock_llm.bind_tools.assert_called_with(tools)

        # Check if invoke was called
        mock_llm_with_tools.ainvoke.assert_called()

        # Verify telemetry was updated
        mock_telemetry.update_counter.assert_called_with(
            "agent.invocations.total", attributes={"agent": "common"}
        )

    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    async def test_supervisor_node(self, mock_settings, mock_llm_factory):
        """Test that supervisor_node returns the correct next_worker."""
        # Setup
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_llm_factory.return_value = mock_llm

        # Create mock requirements that behave like Pydantic models
        req1 = MagicMock()
        req1.model_dump.return_value = {"id": 1, "instruction": "Requirement 1", "assigned_agent": "agent1"}
        req1.id = 1
        req1.instruction = "Requirement 1"
        req1.assigned_agent = "agent1"

        req2 = MagicMock()
        req2.model_dump.return_value = {"id": 2, "instruction": "Requirement 2", "assigned_agent": "agent2"}
        req2.id = 2
        req2.instruction = "Requirement 2"
        req2.assigned_agent = "agent2"

        # Mock the structured output (SupervisorAgentOutput)
        mock_output = MagicMock(
            spec=SupervisorAgentPlanOutput,
            next_worker=MagicMock(value="network_agent"),
            response_to_user="Response for user",
            detected_requirements=[req1, req2],
            instructions_for_worker="Instructions for worker",
            current_step_id=1,
            current_step_status="pending",
            skipped_step_ids=[],
        )
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_output)

        state = {
            "messages": [HumanMessage(content="Network issue")],
            "user_id": "test_user",
            "response_metadata": {},
        }

        # Execute
        result = await supervisor_node(state)

        # Verify
        self.assertEqual(result["next_worker"], "network_agent")

    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")
    async def test_system_node(self, mock_telemetry, mock_settings, mock_llm_factory):
        """Test that system_node correctly invokes the LLM with tools."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_factory.return_value = mock_llm

        expected_response = AIMessage(content="System response")
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=expected_response)

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
        result = await system_node(state, tools)

        # Verify
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], expected_response)
        self.assertIn("system", result["response_metadata"]["nodes"])

        # Check if tools were bound
        mock_llm.bind_tools.assert_called_with(tools)

        # Check if invoke was called
        mock_llm_with_tools.ainvoke.assert_called()

        # Verify telemetry
        mock_telemetry.update_counter.assert_called_with(
            "agent.invocations.total", attributes={"agent": "system"}
        )

    def test_route_workflow(self):
        """Test the routing logic from the supervisor."""
        dummy_msg = HumanMessage(content="test")
        self.assertEqual(
            route_workflow({"next_worker": "network_agent", "messages": [dummy_msg]}), "network_agent"
        )
        self.assertEqual(
            route_workflow({"next_worker": "common_agent", "messages": [dummy_msg]}), "common_agent"
        )

    def test_route_workflow_unknown(self):
        """Test routing with an unknown worker."""
        dummy_msg = HumanMessage(content="test")
        self.assertEqual(
            route_workflow({"next_worker": "unknown", "messages": [dummy_msg]}), "unknown"
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

    def test_detect_tool_loop_different_args(self):
        """Test that same tool with different args is not a loop."""
        call_a = {"name": "tool_a", "args": {"x": 1}, "id": "1"}
        call_b = {"name": "tool_a", "args": {"x": 2}, "id": "2"}
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[call_a]),
                ToolMessage(content="res", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[call_b]),
                ToolMessage(content="res", tool_call_id="2", name="tool_a"),
                AIMessage(content="", tool_calls=[call_a]),
            ]
        }
        self.assertFalse(detect_tool_loop(state))

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

class TestEnforcePlan(unittest.TestCase):
    def setUp(self):
        # Mock the decision object (SupervisorAgentOutput)
        self.decision = MagicMock()
        self.decision.next_worker = "end"
        self.decision.current_step_status = "completed"
        self.decision.response_to_user = "Task finished."
        self.decision.instructions_for_worker = ""
        self.decision.skipped_step_ids = []

        # Standard requirements setup
        self.requirements = [
            {"id": 1, "instruction": "Step 1", "assigned_agent": "agent1"},
            {"id": 2, "instruction": "Step 2", "assigned_agent": "agent2"},
            {"id": 3, "instruction": "Step 3", "assigned_agent": "agent3"},
        ]

    def test_enforce_plan_no_pending_steps(self):
        """Should not modify decision if all steps are completed."""
        steps_status = ["completed", "completed", "completed"]
        enforce_plan(self.decision, self.requirements, steps_status)
        self.assertEqual(self.decision.next_worker, "end")

    def test_enforce_plan_pending_step_explicit(self):
        """Should redirect to supervisor if a step is explicitly marked 'pending'."""
        # Step 2 is pending
        steps_status = ["completed", "pending", "pending"]
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "supervisor")
        self.assertIn("pending steps", self.decision.instructions_for_worker)
        self.assertIn("Step 2", self.decision.instructions_for_worker)

    def test_enforce_plan_pending_step_implicit_length(self):
        """Should redirect to supervisor if steps_status list is shorter than requirements."""
        # Only Step 1 is recorded, Step 2 and 3 are implicitly pending
        steps_status = ["completed"]
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "supervisor")
        self.assertIn("pending steps", self.decision.instructions_for_worker)
        # Should identify the first missing step (index 1 -> Step 2)
        self.assertIn("Step 2", self.decision.instructions_for_worker)

    def test_enforce_plan_abort_on_failure(self):
        """Should allow ending if the current step failed (abort plan)."""
        self.decision.current_step_status = "failed"
        # Even though steps are pending, failure should allow exit
        steps_status = ["completed", "pending"]
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "end")

    def test_enforce_plan_skip_if_asking_approval(self):
        """Should allow ending if supervisor is asking for user approval."""
        steps_status = ["completed", "pending"]
        self.decision.response_to_user = "Please approve the next step before I continue."
        
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "end")

    def test_enforce_plan_skip_if_next_agent_human(self):
        """Should allow ending if the next pending step is assigned to 'human'."""
        steps_status = ["completed", "pending"]
        # Modify requirement 2 to be human
        self.requirements[1]["assigned_agent"] = "human"
        
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "end")

    def test_enforce_plan_not_ending(self):
        """Should do nothing if supervisor is not trying to end."""
        self.decision.next_worker = "agent2"
        steps_status = ["completed", "pending"]
        
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "agent2")

    def test_enforce_plan_mixed_status_with_skipped(self):
        """Should handle skipped steps correctly."""
        # Step 1 completed, Step 2 skipped, Step 3 pending
        steps_status = ["completed", "skipped", "pending"]
        
        enforce_plan(self.decision, self.requirements, steps_status)
        
        self.assertEqual(self.decision.next_worker, "supervisor")
        self.assertIn("Step 3", self.decision.instructions_for_worker)        

class TestGraphUtilities(unittest.TestCase):
    def test_get_detected_requirements(self):
        # Case 1: Pydantic-like objects
        req1 = MagicMock()
        req1.id = 1
        req1.instruction = "instr 1"
        req1.assigned_agent = MagicMock(value="agent1")
        
        state = {"detected_requirements": [req1]}
        result = get_detected_requirements(state)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["assigned_agent"], "agent1")

        # Case 2: Dicts
        req2 = {"id": 2, "instruction": "instr 2", "assigned_agent": "agent2"}
        state = {"detected_requirements": [req2]}
        result = get_detected_requirements(state)
        self.assertEqual(result[0]["id"], 2)

    def test_update_step_status(self):
        steps_status = ["pending", "pending"]
        decision = MagicMock()
        decision.current_step_id = 1
        decision.current_step_status = "completed"
        decision.skipped_step_ids = [2]

        update_step_status(decision, steps_status)
        
        self.assertEqual(steps_status[0], "completed")
        self.assertEqual(steps_status[1], "skipped")

    def test_update_step_status_invalid_id(self):
        """Test update_step_status with an ID that doesn't exist."""
        steps_status = ["pending"]
        decision = MagicMock()
        decision.current_step_id = 99
        decision.current_step_status = "completed"
        decision.skipped_step_ids = []

        update_step_status(decision, steps_status)
        self.assertEqual(steps_status[0], "pending")

    def test_check_completion(self):
        decision = MagicMock()
        decision.next_worker = "supervisor"
        
        # Case 1: All completed
        existing_reqs = [{"id": 1}, {"id": 2}]
        steps_status = ["completed", "completed"]
        
        check_completion(decision, existing_reqs, steps_status)
        self.assertEqual(decision.next_worker, "end")

        # Case 2: Pending exists
        decision.next_worker = "supervisor"
        steps_status = ["completed", "pending"]
        check_completion(decision, existing_reqs, steps_status)
        self.assertEqual(decision.next_worker, "supervisor")

    def test_check_completion_all_skipped(self):
        decision = MagicMock()
        decision.next_worker = "supervisor"
        existing_reqs = [{"id": 1}]
        steps_status = ["skipped"]
        check_completion(decision, existing_reqs, steps_status)
        self.assertEqual(decision.next_worker, "end")

    def test_filter_tools(self):
        tool1 = MagicMock()
        tool1.name = "allowed_tool"
        tool2 = MagicMock()
        tool2.name = "blocked_tool"
        tools = [tool1, tool2]

        # Allow list
        filtered = filter_tools(tools, ["allowed_tool"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "allowed_tool")

        # Regex
        filtered_regex = filter_tools(tools, ["allowed_.*"])
        self.assertEqual(len(filtered_regex), 1)
        
        # None (all)
        filtered_all = filter_tools(tools, None)
        self.assertEqual(len(filtered_all), 2)

    def test_filter_tools_empty(self):
        tool1 = MagicMock()
        tool1.name = "allowed_tool"
        tools = [tool1]

        # Empty allow list
        filtered = filter_tools(tools, [])
        self.assertEqual(len(filtered), 0)

    def test_contains_sensitive_data(self):
        self.assertTrue(contains_sensitive_data("Here is the api_key: 12345"))
        self.assertTrue(contains_sensitive_data("Bearer token 123"))
        self.assertFalse(contains_sensitive_data("Hello world"))

    def test_contains_sensitive_data_extended(self):
        self.assertTrue(contains_sensitive_data("password = 'secret'"))

    def test_delete_tool_messages(self):
        msgs = [
            ToolMessage(content="res", tool_call_id="1", name="t1"),
            AIMessage(content="thought", tool_calls=[{"id": "1", "name": "t1", "args": {}}], id="ai1"),
            HumanMessage(content="hi")
        ]
        updates = delete_tool_messages(msgs)
        
        # Should have RemoveMessage for ToolMessage
        self.assertIsInstance(updates[0], RemoveMessage)
        self.assertEqual(updates[0].id, msgs[0].id)
        
        # Should have replacement AIMessage
        self.assertIsInstance(updates[1], AIMessage)
        self.assertEqual(updates[1].id, "ai1")
        self.assertEqual(updates[1].tool_calls, [])

class TestAuditorNode(unittest.IsolatedAsyncioTestCase):
    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    async def test_auditor_node_pass(self, mock_settings, mock_llm_factory):
        mock_settings.agent.auditor.enabled = True
        mock_settings.agent.auditor.threshold = 0.8
        
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_factory.return_value = mock_llm
        
        mock_report = MagicMock()
        mock_report.score = 0.9
        mock_report.verifications = []
        mock_structured.ainvoke = AsyncMock(return_value=mock_report)
        
        state = {
            "messages": [HumanMessage(content="req"), AIMessage(content="done")],
            "response_metadata": {}
        }
        
        result = await auditor_node(state)
        self.assertIn("QA Passed", result["messages"][0].content)

    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    async def test_auditor_node_fail(self, mock_settings, mock_llm_factory):
        mock_settings.agent.auditor.enabled = True
        mock_settings.agent.auditor.threshold = 0.8
        
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_factory.return_value = mock_llm
        
        mock_report = MagicMock()
        mock_report.score = 0.5
        mock_verification = MagicMock()
        mock_verification.is_success = False
        mock_verification.missing_information = "Missing X"
        mock_verification.correction_instruction = "Do X"
        mock_report.verifications = [mock_verification]
        mock_structured.ainvoke = AsyncMock(return_value=mock_report)
        
        state = {
            "messages": [HumanMessage(content="req"), AIMessage(content="done")],
            "response_metadata": {}
        }
        
        result = await auditor_node(state)
        self.assertIn("QA REJECTION", result["messages"][0].content)
        self.assertIsNone(result["next_worker"])

    @patch("core.graph.settings")
    async def test_auditor_node_pending_steps(self, mock_settings):
        mock_settings.agent.auditor.enabled = True
        state = {
            "messages": [AIMessage(content="working")],
            "steps_status": ["completed", "pending"],
            "response_metadata": {}
        }
        
        result = await auditor_node(state)
        self.assertIn("QA REJECTION", result["messages"][0].content)
        self.assertIn("plan is incomplete", result["messages"][0].content)