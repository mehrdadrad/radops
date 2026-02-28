"""Module for managing system prompts and agent manifests."""
import logging
import os
import re
import sys
import json
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_community.vectorstores import FAISS

from config.config import settings
from core.llm import embedding_factory
from core.llm import llm_factory

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.getenv("RADOPS_CORE_PROMPTS_DIR")
SYSTEM_AGENTS = [
    {
        "name": "system",
        "description": (
            "Internal system operations manager. "
            "Capabilities: Clear memory (short/long term), manage configuration and secrets (e.g. API keys), "
            "MCP server health/connectivity/status check, list tools on MCP server. "
            "Trigger When: User asks to clear memory, forget conversation, set API keys/secrets, "
            "check MCP server health/connectivity, or list MCP tools. "
            "Differentiation: Use ONLY for internal bot management tasks."
        ),
        "tools": [
            "system__list_mcp_server_tools", 
            "system__list_mcp_servers_health",
            "memory__clear_long_term_memory",
            "memory__clear_short_term_memory",
            "secret__set_user_secrets",
            ],
    },
    {
        "name": "human",
        "description": (
            "The end-user (Human in the loop). "
            "Capabilities: Provide approval, confirmation, or missing configuration parameters. "
            "Trigger When: You need explicit approval before executing a sensitive action "
            "(e.g., modifying infrastructure, deleting data) or need to ask a clarifying question "
            "*mid-workflow*. "
            "Differentiation: Use human to PAUSE execution for input. Use end to FINISH execution."
        ),
        "tools": [],
    },
]


def _load_prompt(filename):
    """Loads a prompt template from a file."""
    if "unittest" in sys.modules:
        return ""

    try:
        if PROMPTS_DIR is None:
            path = Path(filename)
        else:
            path = Path(PROMPTS_DIR) / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        logger.error("Prompt file not found: %s", path)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to load prompt from %s: %s", filename, e)
        sys.exit(1)


def generate_agent_manifest(
    agent_name, prompt_text_file, llm_profile
):
    """
    Generates a structured manifest for an agent by analyzing its system prompt.
    Uses caching to avoid re-generation if the file hasn't changed.
    """
    cache_file = None
    try:
        if os.path.exists(prompt_text_file):
            cache_dir = Path(".cache")
            cache_dir.mkdir(exist_ok=True)
            mtime = int(os.path.getmtime(prompt_text_file))
            cache_file = cache_dir / f"{Path(prompt_text_file).name}_{mtime}"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Cache error: %s", e)

    try:
        with open(prompt_text_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Couldn't open file: %s, error: %s", prompt_text_file, e)
        return ""

    logger.info("Generating %s subprompt ...", agent_name)
    prompt = (
        f"Analyze the system prompt for the '{agent_name}' agent below. "
        "Provide a structured summary exactly in the following format:\n"
        "- **Role:** [A concise title and role description]\n"
        "   - **Capabilities:** [Comma-separated list of key capabilities]\n"
        "   - **Technical Keywords:** [List of specific technical terms, protocols, or acronyms this agent handles]\n"
        "   - **Trigger When:** [Specific user intents or keywords]\n"
        "   - **Differentiation:** [When to use this agent over others]\n\n"
        f"System Prompt:\n{prompt_text}"
    )
    content = llm_factory(llm_profile, agent_name="system_manifest").invoke(prompt).content

    if cache_file:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to write cache: %s", e)

    return content

def build_agent_registry(tools: Sequence[BaseTool]):
    """Builds the agent registry by registering available agents including prompt and tools."""
    docs = []
    available_skills = get_available_skills()

    # system agents
    for agent in SYSTEM_AGENTS:
        docs.append(
            Document(
                page_content=agent["description"],
                metadata={"agent_name": agent["name"], "tools": agent["tools"]},
            )
        )

    # dynamic agents
    for agent_name, agent_config in settings.agent.profiles.items():
        prompt_text = generate_agent_manifest(
            agent_name,
            agent_config.system_prompt_file,
            agent_config.manifest_llm_profile or settings.llm.default_profile,  # pylint: disable=no-member
        )

        # Filter tools for this agent
        agent_tools = []
        if agent_config.allow_tools:
            for tool in tools:
                for pattern in agent_config.allow_tools:
                    if re.fullmatch(pattern, tool.name):
                        agent_tools.append(tool)
                        break

        if agent_tools:
            prompt_text += "\n\n### Available Tools\n"
            for tool in agent_tools:
                prompt_text += f"- {tool.name}: {tool.description}\n"

        agent_skills = []
        if agent_config.allow_skills:
            for skill in available_skills:
                for pattern in agent_config.allow_skills:
                    categories = skill["metadata"].get("category") or []
                    if isinstance(categories, str):
                        categories = [categories]

                    if re.fullmatch(pattern, skill["name"]) or any(
                        re.fullmatch(pattern, c) for c in categories
                    ):
                        agent_skills.append(skill)
                        break

        if agent_skills:
            prompt_text += "\n\n### Available Skills\n"
            for skill in agent_skills:
                prompt_text += f"- {skill['name']} (Path: '{skill['path']}'): {skill['description']}\n"

        docs.append(
            Document(
                page_content=prompt_text,
                metadata={
                    "agent_name": agent_name,
                    "tools": [t.name for t in agent_tools],
                    "skills": agent_skills,
                },
            )
        )

    embedding = embedding_factory(settings.discovery.embedding_profile) 

    return FAISS.from_documents(docs, embedding)

def build_skill_registry():
    """
    Builds a registry of skills indexed by their description and associated with agents.
    """
    docs = []
    available_skills = get_available_skills()

    for agent_name, agent_config in settings.agent.profiles.items():
        if not agent_config.allow_skills:
            continue

        for skill in available_skills:
            allowed = False
            for pattern in agent_config.allow_skills:
                categories = skill["metadata"].get("category") or []
                if isinstance(categories, str):
                    categories = [categories]

                if re.fullmatch(pattern, skill["name"]) or any(
                    re.fullmatch(pattern, c) for c in categories
                ):
                    allowed = True
                    break

            if allowed:
                docs.append(
                    Document(
                        page_content=(
                            f"{skill['name']}: {skill['description']}, "
                            f"path: {skill['path']}"
                        ),
                        metadata={"agent_name": agent_name, **skill},
                    )
                )

    if not docs:
        return None

    embedding = embedding_factory(settings.discovery.embedding_profile)
    return FAISS.from_documents(docs, embedding)

def get_available_skills() -> list[dict]:
    """Scans the skills directory and returns a list of available skills."""
    try:
        # Resolve project root relative to this file: src/prompts/system.py
        project_root = Path(__file__).resolve().parents[2]
        skills_dir = project_root / "skills"
        
        if not skills_dir.exists():
            return []

        # Compile regex once for performance
        input_pattern = re.compile(r"## Input(.*?)(?=\n##|\Z)", re.DOTALL)
        skills_list = []

        for root, _, files in os.walk(skills_dir):
            for file in files:
                if not file.endswith(".md"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if not content.startswith("---"):
                        continue

                    parts = content.split("---", 2)
                    if len(parts) < 3:
                        continue

                    frontmatter = parts[1]
                    body = parts[2]
                    
                    name = "Unknown"
                    description = ""
                    metadata = {}
                    
                    for line in frontmatter.splitlines():
                        line = line.strip()
                        if not line or ":" not in line:
                            continue
                        
                        key, val = line.split(":", 1)
                        key = key.strip()
                        val = val.strip()

                        if key == "name":
                            name = val
                        elif key == "description":
                            description = val.strip('"')
                        elif key == "metadata":
                            try:
                                metadata = json.loads(val.strip("'\""))
                            except json.JSONDecodeError:
                                pass    
                    
                    # Extract inputs
                    inputs = []
                    input_match = input_pattern.search(body)
                    if input_match:
                        inputs = [
                            line.strip() 
                            for line in input_match.group(1).splitlines() 
                            if line.strip().startswith("-")
                        ]

                    rel_path = str(file_path.relative_to(skills_dir))
                    skills_list.append({
                        "name": name, 
                        "path": rel_path, 
                        "description": description, 
                        "inputs": inputs, 
                        "metadata": metadata
                    })
                except Exception as e:
                    logger.warning("Error parsing skill %s: %s", file, e)
        
        return skills_list
    except Exception as e:
        logger.error("Failed to load skills: %s", e)
        return []

def get_system_capabilities() -> str:
    """
    Generates a dynamic string of system capabilities based on registered agents.
    Used for guardrails to determine relevance.
    """
    capabilities = ["conversation history", "internal system operations"]

    for _, agent_config in settings.agent.profiles.items():  # pylint: disable=no-member
        if agent_config.description:
            capabilities.append(agent_config.description)

    return ", ".join(capabilities)

def _build_guardrails_prompt():
    """Builds the guardrails system prompt."""
    prompt_template = _load_prompt(settings.agent.guardrails.prompt_file)
    return prompt_template.format(agent_capabilities=get_system_capabilities())

def _build_supervisor_prompt():
    """Builds the supervisor system prompt."""
    prompt_template = _load_prompt(settings.agent.supervisor.prompt_file)

    workers_prompts = ""
    idx = 3
    for agent_name, agent_config in settings.agent.profiles.items():  # pylint: disable=no-member
        agent_manifest = generate_agent_manifest(
            agent_name,
            agent_config.system_prompt_file,
            agent_config.manifest_llm_profile or settings.llm.default_profile,  # pylint: disable=no-member
        )

        workers_prompts += f"""
{idx}. **{agent_name.replace('_', ' ').title()}** (`{agent_name}`):
   {agent_manifest}
"""
        idx += 1

    general_instructions = (
        "- **Specificity Rule:** Always prefer a specialized agent over a general agent "
        "if the request involves technical details.\n"
    )

    discovery_mode = settings.agent.supervisor.discovery_mode
    if discovery_mode != "prompt":
        general_instructions += (
            "- **Agent Discovery:** You MUST use the `system__agent_discovery_tool` "
            "to identify the correct agent for the task. Pass the task description to the tool "
            "to get the recommended agent.\n"
            "- **Constraint:** You are allowed to call `system__agent_discovery_tool` ONLY ONCE per request. "
            "If the tool does not return a suitable agent, do NOT retry with different queries.\n"
            "- If the tool returns 'unavailable', you MUST inform the user that no suitable agent was found "
            "to handle their request and set `next_worker` to 'end'.\n"
        )
    else:
        for agent_name, agent_config in settings.agent.profiles.items():  # pylint: disable=no-member
            if agent_config.description:
                general_instructions += (
                    f"- If the user request matches {agent_config.description}, "
                    f"route to `{agent_name}`.\n"
                )

    return prompt_template.format(
        workers_prompts=workers_prompts,
        general_instructions=general_instructions,
    )

SUPERVISOR_PROMPT = _build_supervisor_prompt()
GUARDRAILS_PROMPT = _build_guardrails_prompt()

PLATFORM_PROMPT = """
You are the System Agent. You are responsible for internal system operations.
Only use the tools explicitly requested by the user or strictly necessary for the task.
Do not guess or run tools preemptively.
Do not set secrets unless explicitly asked.
Once the tool execution is complete and successful, DO NOT reply with text. Instead, use the
'system__submit_work' tool to report completion.
"""

AUDITOR_PROMPT = _load_prompt(settings.agent.auditor.prompt_file)

EXTENSION_PROMPT = """
### User Context
- The user you are speaking with has the user_id: {user_id} which will be automatically injected
  into any tool that requires it.
- When appropriate, you MUST address the user by their `user_id`.

### Supervisor Instructions
- You are a worker agent. You receive tasks from the Supervisor.
- Focus ONLY on the specific instruction provided by the Supervisor in the last message.
- Ignore other parts of the conversation history (e.g., previous user requests) that are not relevant to the current instruction.

### Reporting Results
- When using `system__submit_work`, return `success` and `failure_reason` if applicable.
- **DO NOT** say "The task is complete" or "I have finished the request". The Supervisor tracks the overall progress.
- **Success Criteria:** If a tool runs successfully but returns no data (e.g., "No instances found", "Empty list"), this is still a **SUCCESS**. Set `success=True`. Only set `success=False` if the tool actually failed (e.g., error, exception, timeout).
"""

SUMMARIZATION_PROMPT = """
You are an expert technical writer. Your task is to summarize the conversation history in under {max_summary_tokens} tokens.
### Conversation to Summarize:
{conversation_history}
"""
