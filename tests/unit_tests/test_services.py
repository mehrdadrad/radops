import unittest
import json
from unittest.mock import MagicMock, patch, mock_open, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage

from services.learning.recorder import AdaptiveLearningEngine
from services.guardrails.guardrails import guardrails
from services.telemetry.telemetry import telemetry

class TestAdaptiveLearningEngine(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "test_dataset.jsonl"
        self.engine = AdaptiveLearningEngine(dataset_path=self.dataset_path)

    def test_log_interaction(self):
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there")
        ]
        summary = "A brief chat."
        
        m = mock_open()
        with patch("builtins.open", m):
            self.engine.log_interaction(messages, summary=summary)
            
        m.assert_called_with(self.dataset_path, "a", encoding="utf-8")
        handle = m()
        
        # Verify write
        args, _ = handle.write.call_args
        written_data = args[0]
        data = json.loads(written_data)
        
        if "summary" in data:
            self.assertEqual(data["summary"], summary)
        self.assertEqual(len(data["messages"]), 3)
        # Check message serialization (assuming standard role mapping)
        self.assertEqual(data["messages"][1]["role"], "user")
        self.assertEqual(data["messages"][1]["content"], "Hello")
        self.assertEqual(data["messages"][2]["role"], "assistant")
        self.assertEqual(data["messages"][2]["content"], "Hi there")

    def test_log_interaction_no_summary(self):
        messages = [HumanMessage(content="Hi")]
        m = mock_open()
        with patch("builtins.open", m):
            self.engine.log_interaction(messages, summary=None)
        
        handle = m()
        args, _ = handle.write.call_args
        data = json.loads(args[0])
        self.assertIsNone(data.get("summary"))

class TestGuardrails(unittest.IsolatedAsyncioTestCase):
    @patch("services.guardrails.guardrails.llm_factory")
    @patch("services.guardrails.guardrails.settings")
    async def test_guardrails_disabled(self, mock_settings, mock_llm_factory):
        mock_settings.agent.guardrails.enabled = False
        state = {"messages": [], "user_id": "test_user"}
        
        result = guardrails(state)
        # Should default to continue if disabled
        self.assertEqual(result["end_status"], "continue")
        mock_llm_factory.assert_not_called()

    @patch("builtins.open", new_callable=mock_open, read_data="system prompt")
    @patch("services.guardrails.guardrails.llm_factory")
    @patch("services.guardrails.guardrails.settings")
    async def test_guardrails_safe(self, mock_settings, mock_llm_factory, mock_file):
        mock_settings.agent.guardrails.enabled = True
        mock_settings.agent.guardrails.llm_profile = "default"
        mock_settings.agent.guardrails.prompt_file = "prompt.txt"
        
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_factory.return_value = mock_llm
        
        # Mock output
        mock_decision = MagicMock()
        mock_decision.action = "continue"
        mock_structured.invoke.return_value = mock_decision
        
        state = {"messages": [HumanMessage(content="Hello")], "user_id": "test_user"}
        
        result = guardrails(state)
        self.assertEqual(result["end_status"], "continue")

    @patch("builtins.open", new_callable=mock_open, read_data="system prompt")
    @patch("services.guardrails.guardrails.llm_factory")
    @patch("services.guardrails.guardrails.settings")
    async def test_guardrails_unsafe(self, mock_settings, mock_llm_factory, mock_file):
        mock_settings.agent.guardrails.enabled = True
        mock_settings.agent.guardrails.llm_profile = "default"
        mock_settings.agent.guardrails.prompt_file = "prompt.txt"
        
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_factory.return_value = mock_llm
        
        class MockOutput(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self
            def model_dump(self):
                return self
            def dict(self):
                return self

        mock_decision = MockOutput(action="block", reason="Unsafe content", reasoning="Unsafe content", is_safe=False)
        mock_structured.invoke.return_value = mock_decision
        
        state = {"messages": [HumanMessage(content="Bad content")], "user_id": "test_user", "response_metadata": {}}
        
        result = guardrails(state)
        self.assertEqual(result["end_status"], "end")

class TestTelemetry(unittest.TestCase):
    def test_telemetry_methods(self):
        # Verify methods exist and run without error
        telemetry.update_counter("test_counter", 1, {"env": "test"})
        telemetry.update_histogram("test_hist", 0.5, {"env": "test"})