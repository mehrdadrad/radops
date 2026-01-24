"""Module for managing system prompts and agent manifests."""
import logging
import os
import re
import sys
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
    for agent_name, agent_config in settings.agent.profiles.items():
        prompt_text = ""
        try:
            prompt_file = agent_config.system_prompt_file
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Couldn't open file: %s, error: %s", prompt_file, e)
            sys.exit(1)

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
        
        docs.append( 
            Document(
                page_content=prompt_text, 
                metadata={"agent_name": agent_name}
            )
        )

    embedding = embedding_factory("openai-embedding-small") 

    return FAISS.from_documents(docs, embedding)

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
