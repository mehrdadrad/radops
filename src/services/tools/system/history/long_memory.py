"""Tools for managing long-term memory."""
import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from core.memory import get_mem0_client

class MemoryClearInput(BaseModel):
    """Input schema for clearing long-term memory."""
    user_id: Annotated[str, InjectedState("user_id")] = Field(
        description="The unique identifier for the current user."
    )

logger = logging.getLogger(__name__)

@tool(args_schema=MemoryClearInput)
async def memory__clear_long_term_memory(user_id: str):
    """Clears the long-term memory for a given user.

    Args:
        user_id: The ID of the user.
    """
    try:
        m = await get_mem0_client()
        await m.delete_all(user_id=user_id)
        logger.info("Memory cleared for user: %s", user_id)
        return "Memory cleared"
    except Exception as e:      # pylint: disable=broad-exception-caught
        logger.error("Error clearing memory: %s for user: %s", e, user_id)
        return "Error clearing memory"
    