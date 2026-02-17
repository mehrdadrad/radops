import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.services.tools.system.system.system import (
    create_agent_discovery_tool,
    create_mcp_server_health_tool,
    create_mcp_server_tools_tool,
    system__submit_work
)

class TestSystemTools(unittest.TestCase):
    def test_system_submit_work(self):
        res_success = system__submit_work.invoke({"success": True})
        self.assertIn("Completed Successfully", res_success)
        
        res_fail = system__submit_work.invoke({"success": False, "failure_reason": "error"})
        self.assertIn("Failed", res_fail)
        self.assertIn("error", res_fail)

    def test_mcp_server_health_tool(self):
        # Mock clients
        client1 = MagicMock()
        client1.name = "server1"
        client1.session = True
        client1._running = True
        client1.tools = [1, 2]
        
        client2 = MagicMock()
        client2.name = "server2"
        client2.session = None # Disconnected
        client2._running = True
        client2.tools = []

        tool = create_mcp_server_health_tool([client1, client2])
        result = tool.invoke({})
        
        self.assertIn("server1: Healthy (2 tools)", result)
        self.assertIn("server2: Disconnected", result)

    def test_mcp_server_health_tool_empty(self):
        tool = create_mcp_server_health_tool([])
        result = tool.invoke({})
        self.assertEqual(result, "No MCP servers configured.")

    def test_mcp_server_tools_tool(self):
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "desc1"
        
        client1 = MagicMock()
        client1.name = "server1"
        client1.tools = [tool1]
        
        tool = create_mcp_server_tools_tool([client1])
        
        # Test finding server
        result = tool.invoke({"server_name": "server1"})
        self.assertIn("Tools for server 'server1'", result)
        self.assertIn("tool1: desc1", result)
        
        # Test server not found
        result_not_found = tool.invoke({"server_name": "unknown"})
        self.assertIn("not found", result_not_found)
        self.assertIn("Available servers: server1", result_not_found)

    @patch("src.services.tools.system.system.system.build_agent_registry")
    @patch("src.services.tools.system.system.system.settings")
    def test_agent_discovery_tool(self, mock_settings, mock_build_registry):
        # Setup settings
        mock_settings.agent.supervisor.discovery_threshold = 0.5
        
        # Setup vector store mock
        mock_db = MagicMock()
        mock_build_registry.return_value = mock_db
        
        # Mock search results
        # Doc 1: Good match
        doc1 = Document(page_content="desc1", metadata={"agent_name": "agent1", "tools": ["t1", "t2"]})
        # Doc 2: Bad match (score > threshold)
        doc2 = Document(page_content="desc2", metadata={"agent_name": "agent2", "tools": []})
        
        # similarity_search_with_score returns list of (doc, score)
        mock_db.similarity_search_with_score.return_value = [
            (doc1, 0.1), # Good score (lower is better)
            (doc2, 0.9)  # Bad score
        ]
        
        tool = create_agent_discovery_tool([])
        result = tool.invoke({"queries": ["task1"]})
        
        # Check output format
        self.assertIn("Recommended Agents", result)
        self.assertIn("'agent1'", result)
        self.assertIn("Tools: t1, t2", result) # Check if tools are listed
        self.assertIn("Score: 0.10", result)
        
        # Agent 2 should not be in recommended list because score 0.9 > 0.5
        self.assertNotIn("'agent2'", result)

    @patch("src.services.tools.system.system.system.build_agent_registry")
    @patch("src.services.tools.system.system.system.settings")
    def test_agent_discovery_tool_no_match(self, mock_settings, mock_build_registry):
        mock_settings.agent.supervisor.discovery_threshold = 0.5
        mock_db = MagicMock()
        mock_build_registry.return_value = mock_db
        
        # All results above threshold
        doc1 = Document(page_content="desc1", metadata={"agent_name": "agent1"})
        mock_db.similarity_search_with_score.return_value = [(doc1, 0.8)]
        
        tool = create_agent_discovery_tool([])
        result = tool.invoke({"queries": ["task1"]})
        
        self.assertIn("Agent: 'end|agent1'", result)
        self.assertIn("Score 0.80 > 0.5", result)

    @patch("src.services.tools.system.system.system.build_agent_registry")
    @patch("src.services.tools.system.system.system.settings")
    def test_agent_discovery_tool_empty_search(self, mock_settings, mock_build_registry):
        mock_settings.agent.supervisor.discovery_threshold = 0.5
        mock_db = MagicMock()
        mock_build_registry.return_value = mock_db
        
        mock_db.similarity_search_with_score.return_value = []
        
        tool = create_agent_discovery_tool([])
        result = tool.invoke({"queries": ["task1"]})
        
        self.assertIn("Agent: 'end' (No match)", result)