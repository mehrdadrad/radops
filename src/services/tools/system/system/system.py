import logging
from typing import Any, List, Sequence
from langchain_core.tools import tool
from core.state import WorkerAgentOutput
from langchain_core.tools import BaseTool
from prompts.system import build_agent_registry

logger = logging.getLogger(__name__)


@tool(args_schema=WorkerAgentOutput)
def system__submit_work(success: bool, failure_reason: str | None = None):
    """Submit the final status of the work to the supervisor."""
    if success:
        return "Task Completed Successfully."
    return f"Task Failed. Reason: {failure_reason}."

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

def create_agent_discovery_tool(tools: Sequence[BaseTool]):
    """
    Creates a tool to find agents.
    """
    db = build_agent_registry(tools)
    @tool
    def system__agent_discovery_tool(queries: list[str]):
        """
        Analyzes task steps to identify the appropriate agent for each step.

        This tool takes a list of task descriptions (queries) and returns the name of the
        agent best suited to handle each step, assisting the supervisor in planning.
        """
        print("******* TOOOL CALLED ******", queries)
        agents = []
        for query in queries:
            result = db.similarity_search(query, k=1)
            if result:
                agents.append(result[0].metadata['agent_name'])
            else:
                agents.append("unknown")

        return agents        

    return system__agent_discovery_tool
                 