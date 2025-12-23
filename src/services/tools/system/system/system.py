import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def system__escalate_to_supervisor(reason: str, data: str):
    """
    Call this tool ONLY when you cannot solve the problem or need to hand off 
    to a different agent. This will return control to the Supervisor.

    Args:
        reason: A clear instruction for the next agent (e.g., "Please create a ticket"). Do NOT state the action is already done.
        data: The information collected by the current agent or relevant context to pass along.
    """

    logger.info("Escalating to supervisor")

    return (
        f"ESCALATION REQUEST: {reason}\n"
        f"DATA COLLECTED: {data}\n\n"
        "NOTE TO NEXT AGENT: The previous agent has collected this data. You are now responsible for performing the requested action (e.g., creating a ticket) using this data immediately."
    )