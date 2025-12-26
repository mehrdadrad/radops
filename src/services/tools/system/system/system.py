import logging
from typing import Any, List
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def system__escalate_to_supervisor(reason: str):
    """
    Call this tool ONLY when you cannot solve the problem or need to hand off 
    to a different agent. This will return control to the Supervisor.

    Args:
        reason: Do NOT state the action is already done. 
    """

    logger.info("Escalating to supervisor")

    return (
        f"ESCALATION REQUEST: {reason}\n"
        "NOTE TO NEXT AGENT: The previous agent has collected data. You are now responsible for performing the requested action"
    )

def create_mcp_server_health_tool(mcp_clients: List[Any]):
    """
    Creates a tool to check the health of MCP servers.
    """
    @tool
    def system__list_mcp_servers_health():
        """
        Checks the connectivity and health status of all registered MCP servers.
        """
        if not mcp_clients:
            return "No MCP servers configured."

        results = [f"- {c.name}: {'Healthy' if c.session and c._running else 'Disconnected'} ({len(c.tools)} tools)" for c in mcp_clients]
        return "\n".join(results)

    return system__list_mcp_servers_health