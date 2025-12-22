import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def system__escalate_to_supervisor(reason: str):
    """
    Call this tool ONLY when you cannot solve the problem or need to hand off 
    to a different agent. This will return control to the Supervisor.
    """
    
    logger.info("Escalating to supervisor: %s", reason)

    return f"ESCALATION REQUEST to supervisor: {reason}"