from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config.config import settings


members = {
    k: k for k in list(settings.agent.profiles.keys())
}
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
        description=(
            "The single worker responsible for the very next action. "
            "Select ONLY ONE."
        )
    )
    # The Content: Shown to user
    response_to_user: str = Field(
        description=(
            "A natural language response telling the user about the overall "
            "plan or the next immediate step."
        )
    )
    instructions_for_worker: str = Field(
        description=(
            "The instructions for the selected 'next_worker'. This must only "
            "contain the instructions for the single, immediate next step. "
            "Only include the phrase 'then escalate back to the supervisor' "
            "if you have determined that there are more steps required to "
            "fully answer the user's original request. ensure that no "
            "further action is taken unless explicitly requested"
        )
    )