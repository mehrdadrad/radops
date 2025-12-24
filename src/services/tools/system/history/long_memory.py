from langchain_core.tools import tool
from mem0 import Memory
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
import logging
from typing import Annotated, Optional

class MemoryClearInput(BaseModel):
    user_id: Annotated[str, InjectedState("user_id")] = Field(
        description="The unique identifier for the current user."
    )

logger = logging.getLogger(__name__)

@tool(args_schema=MemoryClearInput)
def memory__clear_long_term_memory(user_id: str):
    """Clears the long-term memory for a given user.

    Args:
        user_id: The ID of the user.
    """
    m = Memory()
    m.delete_all(user_id=user_id)
    logger.info("Memory cleared for user: %s", user_id)