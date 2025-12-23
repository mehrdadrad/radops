"""Module for managing system prompts and agent manifests."""
import logging
import os
from pathlib import Path

from config.config import settings
from core.llm import llm_factory

logger = logging.getLogger(__name__)


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
        "   - **Trigger When:** [Specific user intents or keywords]\n"
        "   - **Differentiation:** [When to use this agent over others]\n\n"
        f"System Prompt:\n{prompt_text}"
    )
    content = llm_factory(llm_profile).invoke(prompt).content

    if cache_file:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to write cache: %s", e)

    return content


SYSTEM_PROMPT = """You are a helpful and professional network assistant.
Your purpose is to help users with network operations by using your available tools.
You can answer questions about router configurations, perform network diagnostics (ASN, ping, trace).
You can ask me about location of IP addresses.
When asked about your identity, introduce yourself as a network assistant. you don't need to explain
Network configuration if it's not requested.
When you use a tool, present the data based on the tool description.
"""


def _build_supervisor_prompt():
    prompt = """
You are the Network Operations Supervisor. Your job is to route user requests to the correct worker. 
You do NOT execute tools or solve problems yourself. You only decide who should handle the task.

### Your Team
"""
    idx = 1
    for agent_name, agent_config in settings.agent.profiles.items():  # pylint: disable=no-member
        agent_manifest = generate_agent_manifest(
                agent_name,
                agent_config.system_prompt_file,
                settings.llm.default_profile,  # pylint: disable=no-member
        )

        prompt += f"""
{idx}. **{agent_name.replace('_', ' ').title()}** (`{agent_name}`):
   {agent_manifest}
"""
        idx += 1
    prompt += """
**Context Handoff (CRITICAL):**
When you route to the *next* agent, do NOT just repeat the original user request. You MUST summarize the previous agent's failure.

### Instructions
"""
    for agent_name, agent_config in settings.agent.profiles.items():  # pylint: disable=no-member
        if agent_config.description:
            prompt += f"- If the user request matches {agent_config.description}, route to `{agent_name}`.\n"

    prompt += """- Route the user's request to the agent whose capabilities best match the intent.
- If a request involves gathering information (e.g., ASN, logs, metrics) AND performing an action (e.g., Jira, GitHub), route to the agent responsible for gathering the information first.
- If an escalated task is already completed, you can finish the task by routing to `end`.
- If the user just says "Hello" or asks a general non-technical question, route to `end`.
- ALWAYS provide a polite `response_to_user` explaining your decision.
"""
    return prompt


SUPERVISOR_PROMPT = _build_supervisor_prompt()

EXTENSION_PROMPT = """
### User Context
- The user you are speaking with has the user_id: {user_id} which will be automatically injected into any tool that requires it.
- When appropriate, you MUST address the user by their `user_id`.
"""
