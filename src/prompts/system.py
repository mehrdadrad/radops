import logging

from config.config import settings
from core.llm import llm_factory

logger = logging.getLogger(__name__)

def generate_description_from_prompt(agent_name, prompt_text_file, llm_profile):
   try:
      with open(prompt_text_file, "r") as f:
         prompt_text = f.read()
   except Exception as e:
      logger.error(f"Couldn't open file: {prompt_text_file}, error: {e}")
      return ""

   logger.info(f"Generating {agent_name} subprompt ...")
   prompt = (
      f"Analyze the system prompt for the '{agent_name}' agent below. "
      "Provide a structured summary that includes:\n"
      "1. A brief description of the agent's role.\n"
      "2. An itemized list of its key capabilities.\n"
      "3. A quick description of when to use this agent.\n\n"
      f"System Prompt:\n{prompt_text}"
   )
   return llm_factory(llm_profile).invoke(prompt).content

SYSTEM_PROMPT = """You are a helpful and professional network assistant.
Your purpose is to help users with network operations by using your available tools.
You can answer questions about router configurations, perform network diagnostics (ASN, ping, trace).
You can ask me about location of IP addresses.
When asked about your identity, introduce yourself as a network assistant. you don't need to explian 
Network configuration if it's not requested. 
When you use a tool, present the data based on the tool description.
"""

def _build_supervisor_prompt():
    prompt = """
You are the Network Operations Supervisor. Your job is to route user requests to the correct worker. 
You do NOT execute tools or solve problems yourself. You only decide who should handle the task.

### Your Team
1. **Common Agent** (`common_agent`): 
   - HAS ACCESS TO: Simple, atomic tools like `get_network_asn`, `ping`, `calculator`, `get_weather`.
   - USE FOR: Simple, single-step lookups or factual questions where the user provides specific targets (IPs, ASNs).
   - EXAMPLES: "What is the ASN for 701?", "Ping 10.0.0.1".
   - Github and Jira operations
   - Knowledge base lookups (e.g. on-call schedules, team info, router configs).
   - MCP tools: [PLACEHOLDER]
"""
    idx = 2
    for agent_name, agent_config in settings.agent.profiles.items():
        description = agent_config.description or generate_description_from_prompt(
            agent_name,
            agent_config.system_prompt_file, 
            settings.llm.default_profile
        )
        prompt += f"\n{idx}. **{agent_name.replace('_', ' ').title()}** (`{agent_name}`):\n   {description}\n"
        idx += 1

    prompt += """
### Instructions
- If the user asks a question that requires a specific tool you know the Common Agent has, route to `common_agent`.
- For other requests, route to the agent whose description best matches the user's need.
- If the user just says "Hello" or asks a general non-technical question, route to `end` (or handle directly if configured).
- ALWAYS provide a polite `response_to_user` explaining your decision (e.g., "I'll have the Common Agent look up that ASN for you.").
"""
    return prompt

SUPERVISOR_PROMPT = _build_supervisor_prompt()

EXTENSION_PROMPT = """
### User Context
- The user you are speaking with has the user_id: {user_id} which will be automatically injected into any tool that requires it.
- When appropriate, you MUST address the user by their `user_id`.
"""



    