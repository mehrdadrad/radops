"""
This module provides a registry for all the tools available to the RadOps assistant.
"""
import asyncio
import logging
import importlib
from typing import List, Any

from langchain_core.tools import BaseTool

from core.mcp_client import MCPClient
from core.vector_store import vector_store_factory

from config.tools import tool_settings as settings
from services.tools.system.config.secrets import secret__set_user_secrets
from services.tools.system.history.history_tools import (
    create_history_deletion_tool,
    create_history_retrieval_tool,
)
from services.tools.system.history.long_memory import memory__clear_long_term_memory
from services.tools.system.kb.kb_tools import create_kb_tools
from services.tools.system.system.system import (
    create_mcp_server_health_tool,
    system__submit_work
)


logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    A registry for all the tools available to the RadOps assistant.
    """

    def __init__(self, checkpointer, skip_initial_sync: bool = False):
        self.checkpointer = checkpointer
        self.vector_store_managers = vector_store_factory(skip_initial_sync=skip_initial_sync)
        mcp_defaults = settings.mcp.model_dump(exclude={"servers"})
        self.mcp_clients = [
            MCPClient(name, {**mcp_defaults, **config})
            for name, config in settings.mcp.servers.items()
            if not config.get("disabled", False)
        ]
        self.weaviate_client = None

    def _import_and_add_tool(self, tools: List[BaseTool], module_path: str, func_name: str):
        try:
            module = importlib.import_module(module_path)
            tool_func = getattr(module, func_name)
            tools.append(tool_func)
            logger.debug("Loaded tool '%s' from '%s'", func_name, module_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load tool '%s' from '%s': %s", func_name, module_path, e)

    def _get_config_value(self, obj: Any, key: str, default: Any = None) -> Any:
        """Helper to get value from dict or object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _load_tools_from_config(self) -> List[BaseTool]:
        """Loads local tools defined in the configuration."""
        tools = []
        # Check if 'local_tools' is defined in settings
        local_tools_config = getattr(settings, "local_tools", None)
        if not local_tools_config:
            return []

        for tool_def in local_tools_config:
            module_path = self._get_config_value(tool_def, "module")
            group_tools = self._get_config_value(tool_def, "tools")

            if group_tools and isinstance(group_tools, list):
                for item in group_tools:
                    func_name = self._get_config_value(item, "function")
                    enabled = self._get_config_value(item, "enabled", True)
                    if enabled and module_path and func_name:
                        self._import_and_add_tool(tools, module_path, func_name)
            else:
                func_name = self._get_config_value(tool_def, "function")
                enabled = self._get_config_value(tool_def, "enabled", True)
                if enabled and module_path and func_name:
                    self._import_and_add_tool(tools, module_path, func_name)

        return tools

    async def get_all_tools(self) -> List[BaseTool]:
        """Gathers and returns all available tools."""
        local_tools = self._load_tools_from_config()
        local_tools = local_tools + [system__submit_work]

        mcp_tools = []
        async def _load_client(client):
            try:
                if not client._running:  # pylint: disable=protected-access
                    await client.start()
                return await client.get_tools()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Failed to load tools from %s: %s", client.name, e)
                return []

        results = await asyncio.gather(*[_load_client(c) for c in self.mcp_clients])
        for tools in results:
            mcp_tools.extend(tools)

        if mcp_tools:
            logger.info("Connected to MCP servers (%s tools found)", len(mcp_tools))

        try:
            dynamic_kb_tools = create_kb_tools(self.vector_store_managers)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("No dynamic knowledge base tools found: %s", e)
            dynamic_kb_tools = []

        return local_tools + mcp_tools + dynamic_kb_tools

    async def get_system_tools(self):
        """Returns a list of system-level tools."""
        system_tools: List[BaseTool] = [
            memory__clear_long_term_memory,
            create_history_deletion_tool(self.checkpointer),
            create_history_retrieval_tool(self.checkpointer),
            create_mcp_server_health_tool(self.mcp_clients),
            secret__set_user_secrets,
            system__submit_work
        ]
        return system_tools


    async def close(self):
        """Closes any open clients."""
        for client in self.mcp_clients:
            await client.stop()

        for manager in self.vector_store_managers:
            manager.close()
