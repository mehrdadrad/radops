import unittest
from unittest.mock import MagicMock, patch
from src.prompts.system import build_agent_registry, generate_agent_manifest

class TestPromptsSystem(unittest.TestCase):
    @patch("src.prompts.system.settings")
    @patch("src.prompts.system.FAISS")
    @patch("src.prompts.system.embedding_factory")
    @patch("src.prompts.system.generate_agent_manifest")
    def test_build_agent_registry(self, mock_gen_manifest, mock_emb_factory, mock_faiss, mock_settings):
        # Setup settings
        mock_settings.agent.profiles = {
            "agent1": MagicMock(
                system_prompt_file="p1.txt",
                manifest_llm_profile="default",
                allow_tools=["tool_a.*"]
            )
        }
        
        # Mock manifest generation
        mock_gen_manifest.return_value = "Agent 1 Description"
        
        # Mock tools
        tool_a = MagicMock()
        tool_a.name = "tool_a_1"
        tool_a.description = "desc a"
        
        tool_b = MagicMock()
        tool_b.name = "tool_b_1" # Should be filtered out
        
        tools = [tool_a, tool_b]
        
        # Execute
        build_agent_registry(tools)
        
        # Verify FAISS.from_documents called
        mock_faiss.from_documents.assert_called_once()
        args, _ = mock_faiss.from_documents.call_args
        docs = args[0]
        
        # Check system agents are present (SYSTEM_AGENTS list in src/prompts/system.py)
        # Usually "system" and "human"
        system_doc = next((d for d in docs if d.metadata["agent_name"] == "system"), None)
        self.assertIsNotNone(system_doc)
        
        # Check dynamic agent
        agent1_doc = next((d for d in docs if d.metadata["agent_name"] == "agent1"), None)
        self.assertIsNotNone(agent1_doc)
        self.assertIn("Agent 1 Description", agent1_doc.page_content)
        self.assertIn("tool_a_1", agent1_doc.metadata["tools"])
        self.assertNotIn("tool_b_1", agent1_doc.metadata["tools"])
        
        # Check if tools description appended to prompt text in page_content
        self.assertIn("Available Tools", agent1_doc.page_content)
        self.assertIn("tool_a_1", agent1_doc.page_content)

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="prompt content")
    @patch("src.prompts.system.os.path.exists", return_value=True)
    @patch("src.prompts.system.llm_factory")
    def test_generate_agent_manifest(self, mock_llm_factory, mock_exists, mock_open):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Manifest Summary"
        mock_llm_factory.return_value = mock_llm
        
        result = generate_agent_manifest("agent1", "path/to/prompt", "profile")
        
        self.assertEqual(result, "Manifest Summary")
        mock_llm.invoke.assert_called()
        # Check prompt sent to LLM contains the read content
        call_args = mock_llm.invoke.call_args[0][0]
        self.assertIn("prompt content", call_args)