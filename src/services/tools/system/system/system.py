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

        results = []
        for c in mcp_clients:
            status = "Healthy" if c.session and c._running else "Disconnected"
            results.append(f"- {c.name}: {status} ({len(c.tools)} tools)")

        return "\n".join(results)

    return system__list_mcp_servers_health

def create_mcp_server_tools_tool(mcp_clients: List[Any]):
    """
    Creates a tool to list tools available on a specific MCP server.
    """
    @tool
    def system__list_mcp_server_tools(server_name: str):
        """
        Lists all available tools for a specific MCP server.
        
        Args:
            server_name: The name of the MCP server to inspect.
        """
        if not mcp_clients:
            return "No MCP servers configured."

        for c in mcp_clients:
            if c.name.lower() == server_name.lower():
                if not c.tools:
                    return f"No tools found for server '{c.name}'."
                
                results = [f"Tools for server '{c.name}':"]
                for t in c.tools:
                    results.append(f"- {t.name}: {t.description}")
                return "\n".join(results)
        
        available = ", ".join([c.name for c in mcp_clients])
        return f"MCP server '{server_name}' not found. Available servers: {available}"

    return system__list_mcp_server_tools

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
            search_results = db.similarity_search_with_score(query, k=2)
            candidates = []
            if search_results:
                for doc, score in search_results:
                    logger.info(
                        "Agent discovery score for query '%s': %f (Agent: %s)",
                        query,
                        score,
                        doc.metadata["agent_name"],
                    )
                    if score <= threshold:
                        candidates.append(
                            f"'{doc.metadata['agent_name']}' (Score: {score:.2f})"
                        )

                if candidates:
                    results.append(
                        f"Query: '{query}' -> Recommended Agents: "
                        f"[{', '.join(candidates)}] (Lower score is a better match)"
                    )
                else:
                    top_doc, top_score = search_results[0]
                    results.append(
                        f"Query: '{query}' -> Agent: 'end|{top_doc.metadata['agent_name']}' "
                        f"(Score {top_score:.2f} > {threshold})"
                    )
            else:
                results.append(f"Query: '{query}' -> Agent: 'end' (No match)")
        
        return "\n".join(results)

    return system__agent_discovery_tool
                 