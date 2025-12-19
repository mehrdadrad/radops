from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    user_id: str
    end_status: Literal["end", "continue"]
    relevant_memories: str
    next_worker: Literal[
        "common_agent",
        "atomic_tool",
        "network_specialist",
        "cloud_specialist",
        "end",
    ]


class SupervisorAgentOutput(BaseModel):
    # The Logic: Hidden from user, used by Graph
    next_worker: Literal[
        "common_agent",
        "atomic_tool",
        "network_specialist",
        "cloud_specialist",
    ] = Field(
        description="The next logical step to take, if it needs to take an action "
    )
    # The Content: Shown to user
    response_to_user: str = Field(
        description="A natural language response telling the user what you are doing."
    )
