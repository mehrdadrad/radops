from typing import List
import logging

from httpx import ConnectError
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from services.tools.system.kb.kb_tools import create_kb_tools

from config.tools import tool_settings as settings
from core.vector_store import vector_store_factory
from services.tools.geoip.geoip import get_geoip_location
from services.tools.checkhost.check_host_net import (
    check_host_ping,
    check_host_http,
    check_host_tcp,
    check_host_dns,
    get_check_host_nodes,
)
from services.tools.system.history.history_tools import (
    create_history_deletion_tool,
    create_history_retrieval_tool,
)
from services.tools.system.config.secrets import set_user_secrets
from services.tools.jira.jira_tools import (
    create_jira_ticket,
    search_jira_issues,
)
from services.tools.lg.lg import verizon_looking_glass, verizon_looking_glass_locations # noqa: E501
from services.tools.github.issue import (
    github_list_issues,
    github_create_issue,
)
from services.tools.github.pull_request import (
    github_create_pull_request,
    github_list_pull_requests,
)
from services.tools.peeringdb.peeringdb import (
    get_asn_peering_info, 
    get_peering_exchange_info,
)
from services.tools.aws.diagnostics import (
    analyze_reachability,
    query_logs,
    check_recent_changes,
    get_ec2_health,
)
from services.tools.aws.ec2 import list_ec2_instances, manage_ec2_instance
from services.tools.aws.troubleshooting import (
    get_cloudformation_stack_events,
    get_target_group_health,
    simulate_iam_policy,
)


logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.vector_store_managers = vector_store_factory()
        self.mcp_client = MultiServerMCPClient(settings.mcp_servers)
        self.weaviate_client = None

    async def get_all_tools(self) -> List[BaseTool]:
        """Gathers and returns all available tools."""
        system_tools: List[BaseTool] = [
            create_history_deletion_tool(self.checkpointer),
            create_history_retrieval_tool(self.checkpointer),
            set_user_secrets,
        ]

        local_tools = [
            get_asn_peering_info,
            get_peering_exchange_info,
            verizon_looking_glass,
            verizon_looking_glass_locations,
            get_geoip_location,
            create_jira_ticket,
            search_jira_issues,
            check_host_ping,
            check_host_http,
            check_host_tcp,
            check_host_dns,
            get_check_host_nodes,
            github_list_issues,
            github_create_issue,
            github_create_pull_request,
            github_list_pull_requests,
            analyze_reachability,
            query_logs,
            check_recent_changes,
            get_ec2_health,
            list_ec2_instances,
            manage_ec2_instance,
            get_cloudformation_stack_events,
            get_target_group_health,
            simulate_iam_policy,
        ]

        try:
            mcp_tools = await self.mcp_client.get_tools()
            if len(mcp_tools) > 0:
                logger.info(f"Connected to MCP servers ({len(mcp_tools)} tools found)")
        except (ConnectError, ExceptionGroup) as e:
            # We need to handle this to prevent the app from crashing.
            if isinstance(e, ExceptionGroup):
                # We only want to suppress connection errors, so we check for them.
                if not any(isinstance(exc, ConnectError) for exc in e.exceptions):
                    raise  # Re-raise if it's not a connection error.

            logger.error(
                f"Could not connect to MCP server(s) to get tools: {e}. "
                "Continuing without them."
            )
            mcp_tools = []

        try:
            dynamic_kb_tools = create_kb_tools(self.vector_store_managers)
        except Exception as e:
            logger.error(f"No dynamic knowledge base tools found: {e}")
            dynamic_kb_tools = []

        return system_tools + local_tools + mcp_tools + dynamic_kb_tools

    async def close(self):
        """Closes any open clients."""
        pass