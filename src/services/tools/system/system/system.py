import logging
from typing import Any, List, Sequence
from langchain_core.tools import tool
from core.state import WorkerAgentOutput
from langchain_core.tools import BaseTool
from config.config import settings
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
        USE THIS TOOL ONLY TO FIND AGENTS.

        This tool takes a list of task descriptions (queries) and returns the name of the
        agent best suited to handle each step.

        Args:
            queries: A list of strings. Each string describes a specific task.
                     Do not pass any other arguments.
        """
        results = []
        # Threshold for similarity score (L2 distance).
        # Lower is better.
        threshold = settings.agent.supervisor.discovery_threshold

        for query in queries:
            result = db.similarity_search_with_score(query, k=1)
            if result:
                doc, score = result[0]
                logger.info("Agent discovery score for query '%s': %f", query, score)
                if score < threshold:
                    results.append(f"Query: '{query}' -> Agent: '{doc.metadata['agent_name']}'")
                else:
                    results.append(f"Query: '{query}' -> Agent: 'end' (Score {score:.2f} > {threshold})")
            else:
                results.append(f"Query: '{query}' -> Agent: 'end' (No match)")

        return "\n".join(results)

    return system__agent_discovery_tool
                 