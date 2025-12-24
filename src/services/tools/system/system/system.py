import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def system__escalate_to_supervisor(reason: str):
    """
    Call this tool ONLY when you cannot solve the problem or need to hand off 
    to a different agent. This will return control to the Supervisor.

    Args:
        reason: Do NOT state the action is already done. 
    """

    logger.info("Escalating to supervisor")

    return (
        f"ESCALATION REQUEST: {reason}\n"
        "NOTE TO NEXT AGENT: The previous agent has collected data. You are now responsible for performing the requested action"
    )