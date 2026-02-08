import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.registry.tools import ToolRegistry

class TestToolRegistry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_checkpointer = MagicMock()
        
        # Patch settings
        self.settings_patcher = patch("src.registry.tools.settings")
        self.mock_settings = self.settings_patcher.start()
        
        # Patch vector_store_factory
        self.vs_factory_patcher = patch("src.registry.tools.vector_store_factory")
        self.mock_vs_factory = self.vs_factory_patcher.start()
        self.mock_vs_manager = MagicMock()
        self.mock_vs_factory.return_value = [self.mock_vs_manager]

        # Patch MCPClient
        self.mcp_client_patcher = patch("src.registry.tools.MCPClient")
        self.mock_mcp_client_cls = self.mcp_client_patcher.start()
        self.mock_mcp_client = MagicMock()
        self.mock_mcp_client_cls.return_value = self.mock_mcp_client
        
        # Patch create_kb_tools
        self.kb_tools_patcher = patch("src.registry.tools.create_kb_tools")
        self.mock_create_kb_tools = self.kb_tools_patcher.start()

    def tearDown(self):
        self.settings_patcher.stop()
        self.vs_factory_patcher.stop()
        self.mcp_client_patcher.stop()
        self.kb_tools_patcher.stop()

    def test_initialization(self):
        # Setup settings for MCP
        self.mock_settings.mcp.model_dump.return_value = {"retry_attempts": 3}
        self.mock_settings.mcp.servers = {
            "server1": {"command": "cmd1"},
            "server2": {"command": "cmd2", "disabled": True}
        }
        
        registry = ToolRegistry(self.mock_checkpointer)
        
        # Check vector store factory call
        self.mock_vs_factory.assert_called_with(skip_initial_sync=False)
        
        # Check MCP client creation (only enabled ones)
        self.mock_mcp_client_cls.assert_called_once()
        args, kwargs = self.mock_mcp_client_cls.call_args
        self.assertEqual(args[0], "server1")
        self.assertEqual(args[1]["command"], "cmd1")
        self.assertEqual(len(registry.mcp_clients), 1)

    def test_initialization_skip_sync(self):
        registry = ToolRegistry(self.mock_checkpointer, skip_initial_sync=True)
        self.mock_vs_factory.assert_called_with(skip_initial_sync=True)

    @patch("src.registry.tools.importlib.import_module")
    def test_load_tools_from_config(self, mock_import):
        # Helper class to simulate Pydantic models or objects
        class ToolConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        conf1 = ToolConfig(module="mod1", function="func1", enabled=True, tools=[])
        conf2 = ToolConfig(module="mod2", tools=[
            ToolConfig(function="func2a", enabled=True),
            ToolConfig(function="func2b", enabled=False)
        ])
        
        self.mock_settings.local_tools = [conf1, conf2]
        
        # Mock import
        mock_module = MagicMock()
        mock_func1 = MagicMock()
        mock_func2a = MagicMock()
        
        mock_import.return_value = mock_module
        mock_module.func1 = mock_func1
        mock_module.func2a = mock_func2a
        
        registry = ToolRegistry(self.mock_checkpointer)
        tools = registry._load_tools_from_config()
        
        self.assertIn(mock_func1, tools)
        self.assertIn(mock_func2a, tools)
        self.assertEqual(len(tools), 2)

    @patch("src.registry.tools.importlib.import_module")
    def test_load_tools_import_error(self, mock_import):
        class ToolConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        conf1 = ToolConfig(module="bad_mod", function="func1", enabled=True, tools=[])
        self.mock_settings.local_tools = [conf1]
        
        mock_import.side_effect = ImportError("Boom")
        
        registry = ToolRegistry(self.mock_checkpointer)
        
        with self.assertLogs("src.registry.tools", level="ERROR") as cm:
            tools = registry._load_tools_from_config()
            self.assertEqual(len(tools), 0)
            self.assertTrue(any("Failed to load tool" in o for o in cm.output))

    async def test_get_all_tools(self):
        registry = ToolRegistry(self.mock_checkpointer)
        
        # Mock local tools
        mock_local_tool = MagicMock()
        registry._load_tools_from_config = MagicMock(return_value=[mock_local_tool])
        
        # Mock MCP tools
        mock_mcp_tool = MagicMock()
        self.mock_mcp_client._running = False
        self.mock_mcp_client.start = AsyncMock()
        self.mock_mcp_client.get_tools = AsyncMock(return_value=[mock_mcp_tool])
        registry.mcp_clients = [self.mock_mcp_client]
        
        # Mock KB tools
        mock_kb_tool = MagicMock()
        self.mock_create_kb_tools.return_value = [mock_kb_tool]
        
        tools = await registry.get_all_tools()
        
        self.assertIn(mock_local_tool, tools)
        self.assertIn(mock_mcp_tool, tools)
        self.assertIn(mock_kb_tool, tools)
        
        # Verify MCP start called
        self.mock_mcp_client.start.assert_called_once()
        
        # Verify system__submit_work is added (it's hardcoded in get_all_tools)
        # local_tools (1) + system__submit_work (1) + mcp (1) + kb (1) = 4
        self.assertEqual(len(tools), 4)

    async def test_get_all_tools_mcp_error(self):
        registry = ToolRegistry(self.mock_checkpointer)
        registry._load_tools_from_config = MagicMock(return_value=[])
        
        self.mock_mcp_client._running = False
        self.mock_mcp_client.start = AsyncMock(side_effect=Exception("Connection failed"))
        registry.mcp_clients = [self.mock_mcp_client]
        
        self.mock_create_kb_tools.return_value = []
        
        with self.assertLogs("src.registry.tools", level="ERROR") as cm:
            tools = await registry.get_all_tools()
            self.assertTrue(any("Failed to load tools from" in o for o in cm.output))
            # Should still have system__submit_work
            self.assertEqual(len(tools), 1) 

    async def test_get_all_tools_kb_error(self):
        registry = ToolRegistry(self.mock_checkpointer)
        registry._load_tools_from_config = MagicMock(return_value=[])
        registry.mcp_clients = []
        
        self.mock_create_kb_tools.side_effect = Exception("KB Error")
        
        with self.assertLogs("src.registry.tools", level="ERROR") as cm:
            tools = await registry.get_all_tools()
            self.assertTrue(any("No dynamic knowledge base tools found" in o for o in cm.output))

    async def test_get_system_tools(self):
        registry = ToolRegistry(self.mock_checkpointer)
        tools = await registry.get_system_tools()
        
        # Check for expected tools
        # We can just check length or non-empty
        self.assertTrue(len(tools) >= 5)

    async def test_close(self):
        registry = ToolRegistry(self.mock_checkpointer)
        registry.mcp_clients = [self.mock_mcp_client]
        self.mock_mcp_client.stop = AsyncMock()
        
        await registry.close()
        
        self.mock_mcp_client.stop.assert_called_once()
        self.mock_vs_manager.close.assert_called_once()
