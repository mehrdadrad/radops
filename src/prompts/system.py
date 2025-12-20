import logging
import os
from pathlib import Path

from config.config import settings
from core.llm import llm_factory

logger = logging.getLogger(__name__)


def generate_agent_manifest(
    agent_name, prompt_text_file, llm_profile
):
    cache_file = None
    try:
        if os.path.exists(prompt_text_file):
            cache_dir = Path(".cache")
            cache_dir.mkdir(exist_ok=True)
            mtime = int(os.path.getmtime(prompt_text_file))
            cache_file = cache_dir / f"{Path(prompt_text_file).name}_{mtime}"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return f.read()
    except Exception as e:
        logger.warning("Cache error: %s", e)

    try:
        with open(prompt_text_file, "r") as f:
            prompt_text = f.read()
    except Exception as e:
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
            with open(cache_file, "w") as f:
                f.write(content)
        except Exception as e:
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
1. **Common Agent** (`common_agent`)
   - **Role:** Tier 1 Support & Tool Runner.
   - **Capabilities:** Atomic tool execution (Ping, ASN lookup), GitHub/Jira management, and Knowledge Base search.
   - **Trigger When:** The user asks for a simple fact, a specific single-step tool execution ("Ping X"), or administrative tasks.
   - **Differentiation:** The default for simple, defined tasks. If the request requires *investigation* or *diagnosis*, do NOT use this agent.
   - **MCP Tools:** [PLACEHOLDER]
"""
    idx = 2
    for agent_name, agent_config in settings.agent.profiles.items():
        agent_manifest = generate_agent_manifest(
                agent_name,
                agent_config.system_prompt_file,
                settings.llm.default_profile,
        )
        
        prompt += f"""
{idx}. **{agent_name.replace('_', ' ').title()}** (`{agent_name}`):
   {agent_manifest}
"""
        idx += 1

    prompt += """
### Instructions
- If the user asks a question that requires a specific tool you know the Common Agent has, route to `common_agent`.
- For other requests, route to the agent whose description best matches the user's need.
- If the user just says "Hello" or asks a general non-technical question, route to `end` (or handle directly if configured).
- ALWAYS provide a polite `response_to_user` explaining your decision (e.g., "I'll have the Common Agent look up that ASN for you.").
"""
    print(prompt)
    return prompt


SUPERVISOR_PROMPT = _build_supervisor_prompt()

EXTENSION_PROMPT = """
### User Context
- The user you are speaking with has the user_id: {user_id} which will be automatically injected into any tool that requires it.
- When appropriate, you MUST address the user by their `user_id`.
"""
