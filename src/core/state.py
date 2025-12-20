from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config.config import settings


members = {k: k for k in ["common_agent"] + list(settings.agent.profiles.keys())}
WorkerEnum = Enum("WorkerEnum", members, type=str)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    user_id: str
    end_status: Literal["end", "continue"]
    relevant_memories: str
    next_worker: str
    response_metadata: dict[str, Any]


class SupervisorAgentOutput(BaseModel):
    # The Logic: Hidden from user, used by Graph
    next_worker: WorkerEnum = Field(
        description="The next logical step to take, if it needs to take an action "
    )
    # The Content: Shown to user
    response_to_user: str = Field(
        description="A natural language response telling the user what you are doing."
    )
