import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from core.graph import create_agent


class TestGraphWithFakeLLM(unittest.IsolatedAsyncioTestCase):
    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")
    async def test_create_agent_conversation_flow(self, mock_telemetry, mock_settings, mock_llm_factory):
        """
        Test create_agent using GenericFakeChatModel to simulate a multi-turn conversation.
        """
        # Define the sequence of messages the Fake LLM should return
        fake_responses = [
            AIMessage(content="Hello! How can I help you?"),
            AIMessage(content="I can certainly check that for you.")
        ]
        
        # Initialize GenericFakeChatModel with the iterator
        fake_model = GenericFakeChatModel(messages=iter(fake_responses))
        
        # Mock the factory to return a mock that returns our fake model when bind_tools is called
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = fake_model
        mock_llm_factory.return_value = mock_llm
        
        # Setup inputs
        tools = [MagicMock()]
        agent_node = create_agent("test_agent", "system prompt", tools)
        
        state = {
            "messages": [HumanMessage(content="Hi")],
            "user_id": "test_user",
            "response_metadata": {}
        }
        
        # First turn
        result1 = await agent_node(state)
        self.assertEqual(result1["messages"][0].content, "Hello! How can I help you?")
        self.assertIn("test_agent", result1["response_metadata"]["nodes"])
        
        # Second turn
        result2 = await agent_node(state)
        self.assertEqual(result2["messages"][0].content, "I can certainly check that for you.")

    @patch("core.graph.llm_factory")
    @patch("core.graph.settings")
    @patch("core.graph.telemetry")
    async def test_create_agent_tool_execution(self, mock_telemetry, mock_settings, mock_llm_factory):
        """
        Test create_agent using GenericFakeChatModel to simulate tool calling.
        """
        # Define a response that includes a tool call
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ping",
                    "args": {"target": "8.8.8.8"},
                    "id": "call_ping_1"
                }
            ]
        )
        
        fake_model = GenericFakeChatModel(messages=iter([tool_call_message]))
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = fake_model
        mock_llm_factory.return_value = mock_llm
        
        tools = [MagicMock()]
        agent_node = create_agent("network_agent", "system prompt", tools)
        
        state = {
            "messages": [HumanMessage(content="Ping 8.8.8.8")],
            "user_id": "test_user",
            "response_metadata": {}
        }
        
        result = await agent_node(state)
        
        message = result["messages"][0]
        self.assertTrue(message.tool_calls)
        self.assertEqual(message.tool_calls[0]["name"], "ping")
        self.assertEqual(message.tool_calls[0]["args"]["target"], "8.8.8.8")
        
        # Verify telemetry was called
        mock_telemetry.update_counter.assert_called_with(
            "agent.invocations.total", attributes={"agent": "network_agent"}
        )