"""
This module provides a registry for all the tools available to the RadOps assistant.
"""
import asyncio
import logging
from typing import List

from langchain_core.tools import BaseTool
from core.mcp_client import MCPClient

from config.tools import tool_settings as settings
from core.vector_store import vector_store_factory
from services.tools.aws.diagnostics import (
    aws__analyze_reachability,
    aws__check_recent_changes,
    aws__get_ec2_health,
    aws__query_logs,
)
from services.tools.aws.ec2 import aws__list_ec2_instances, aws__manage_ec2_instance
from services.tools.aws.network import (
    aws__manage_vpc,
    aws__manage_subnet,
    aws__manage_internet_gateway,
    aws__manage_route_table,
    aws__manage_route,
)
from services.tools.aws.troubleshooting import (
    aws__get_cloudformation_stack_events,
    aws__get_target_group_health,
    aws__simulate_iam_policy,
)
from services.tools.network.checkhost.check_host_net import (
    network__check_host,
    network__get_check_host_nodes,
)
from services.tools.network.geoip.geoip import network__get_geoip_location
from services.tools.github.issue import github_create_issue, github_list_issues
from services.tools.github.pull_request import (
    github_create_pull_request,
    github_list_pull_requests,
)
from services.tools.jira.jira_tools import create_jira_ticket, search_jira_issues
from services.tools.network.lg.lg import (
    network__verizon_looking_glass,
    network__verizon_looking_glass_locations,
)
from services.tools.network.peeringdb.peeringdb import (
    network__get_asn_peering_info,
    network__get_peering_exchange_info,
)
from services.tools.system.config.secrets import set_user_secrets
from services.tools.system.history.history_tools import (
    create_history_deletion_tool,
    create_history_retrieval_tool,
)
from services.tools.system.kb.kb_tools import create_kb_tools

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    A registry for all the tools available to the RadOps assistant.
    """

    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.vector_store_managers = vector_store_factory()
        self.mcp_clients = [
            MCPClient(name, config)
            for name, config in settings.mcp_servers.items()
            if not config.get("disabled", False)
        ]
        self.weaviate_client = None

    async def get_all_tools(self) -> List[BaseTool]:
        """Gathers and returns all available tools."""
        system_tools: List[BaseTool] = [
            create_history_deletion_tool(self.checkpointer),
            create_history_retrieval_tool(self.checkpointer),
            set_user_secrets,
        ]

        local_tools = [
            network__get_asn_peering_info,
            network__get_peering_exchange_info,
            network__verizon_looking_glass,
            network__verizon_looking_glass_locations,
            network__get_geoip_location,
            create_jira_ticket,
            search_jira_issues,
            network__check_host,
            network__get_check_host_nodes,
            github_list_issues,
            github_create_issue,
            github_create_pull_request,
            github_list_pull_requests,
            aws__analyze_reachability,
            aws__query_logs,
            aws__check_recent_changes,
            aws__get_ec2_health,
            aws__list_ec2_instances,
            aws__manage_ec2_instance,
            aws__get_cloudformation_stack_events,
            aws__get_target_group_health,
            aws__simulate_iam_policy,
            aws__manage_vpc,
            aws__manage_subnet,
            aws__manage_internet_gateway,
            aws__manage_route_table,
            aws__manage_route,
        ]

        mcp_tools = []
        async def _load_client(client):
            try:
                if not client._running:
                    await client.start()
                return await client.get_tools()
            except Exception as e:
                logger.error("Failed to load tools from %s: %s", client.name, e)
                return []

        results = await asyncio.gather(*[_load_client(c) for c in self.mcp_clients])
        for tools in results:
            mcp_tools.extend(tools)

        if mcp_tools:
            logger.info("Connected to MCP servers (%s tools found)", len(mcp_tools))

        try:
            dynamic_kb_tools = create_kb_tools(self.vector_store_managers)
        except Exception as e:
            logger.error("No dynamic knowledge base tools found: %s", e)
            dynamic_kb_tools = []

        return system_tools + local_tools + mcp_tools + dynamic_kb_tools

    async def close(self):
        """Closes any open clients."""
        for client in self.mcp_clients:
            await client.stop()