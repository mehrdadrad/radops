import logging
import os
import re
import io
import sys
from contextlib import redirect_stdout
from typing import Any, List, Sequence, Optional
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

        agents_tools = {}
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
                        tool_list = doc.metadata.get("tools", [])
                        skill_list = doc.metadata.get("skills", [])
                        candidates.append(
                            f"'{doc.metadata['agent_name']}' (Score: {score:.2f})"
                        )
                        if doc.metadata["agent_name"] not in agents_tools:
                            agents_tools[doc.metadata["agent_name"]] = {}
                        agents_tools[doc.metadata["agent_name"]]["tools"] = tool_list
                        agents_tools[doc.metadata["agent_name"]]["skills"] = skill_list

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

        output_parts = ["\n".join(results)]
        for agent, data in agents_tools.items():
            tools_str = ", ".join(data.get("tools", []))
            skills_list = data.get("skills", [])
            skills_str = ", ".join([s["name"] for s in skills_list]) if skills_list else ""

            info = f"## Agent {agent}"
            if tools_str:
                info += f"\n- Tools: {tools_str}"
            if skills_str:
                info += f"\n- Skills: {skills_str}"
            output_parts.append(info)

        return "\n\n".join(output_parts)

    return system__agent_discovery_tool

def create_skill_loader_tool(skills_dir: str | None = None):
    """
    Creates a tool to load and register a skill from a SKILL.md file.
    """
    if skills_dir is None:
        # Resolve default skills directory relative to project root
        # This file is at src/services/tools/system/system/system.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, "../../../../../"))
        skills_dir = os.path.join(project_root, "skills")

    # Ensure the directory exists
    if not os.path.exists(skills_dir):
        try:
            os.makedirs(skills_dir)
        except OSError:
            pass

    @tool
    def system__load_skill_from_markdown(file_name: str, variables: Optional[dict] = None):
        """
        Loads and executes a skill definition from a markdown file in the skills directory.

        Args:
            file_name: The relative path to the skill file (e.g., 'ripe-bgp-state/SKILL.md').
            variables: Dictionary of variables to inject (e.g., {'resource': '1.1.1.1'}). Required if the skill expects input.
        """
        file_path = os.path.join(skills_dir, file_name)
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"

            with open(file_path, "r") as f:
                content = f.read()

            # Extract python code block
            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if not code_match:
                return f"No Python code block found in {file_name}."
            
            code = code_match.group(1)

            # Prepare execution environment
            output_capture = io.StringIO()
            exec_globals = {}
            exec_locals = variables if variables else {}

            # Capture stdout
            original_argv = sys.argv
            try:
                if variables:
                    sys.argv = ["skill_script"] + [str(v) for v in variables.values()]
                with redirect_stdout(output_capture):
                    try:
                        exec(code, exec_globals, exec_locals)
                    except Exception as e:
                        print(f"Error executing skill: {e}")
            finally:
                sys.argv = original_argv
            
            return output_capture.getvalue()

        except Exception as e:
            return f"Failed to read or execute skill file: {e}"

    return system__load_skill_from_markdown
                 