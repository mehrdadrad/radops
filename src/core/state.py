from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config.config import settings


members = {
    k: k for k in list(settings.agent.profiles.keys())
}
# Add a way to end the conversation
members["system"] = "system"
members["end"] = "end"
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
            "Select ONLY ONE. Select 'end' if you have gathered enough "
            "information to fully answer the user's question."
        )
    )
    # The Content: Shown to user
    response_to_user: str = Field(
        description=(
            "Communication to the user. Follow these strict rules:"
            "1. **Planning Step:** explain how do you want to break up the task and what agent will be responsible for each step. **DO NOT** repeat the data found by the previous agent."
            "2. **Intermediate Step:** If you are routing to a worker (Network/Cloud), provide a BRIEF status update only (e.g., 'ASN info gathered, now checking Cloud'). **DO NOT** repeat the data found by the previous agent."
            "3. **Final Step:** If you are routing to 'end'/'finish', provide the complete, detailed summary of ALL gathered information."
        )
    )
    instructions_for_worker: str = Field(
        description=(
            "The instructions for the selected 'next_worker'. This must only "
            "contain the instructions for the single, immediate next step. "
            "Only include the phrase 'then escalate back to the supervisor' "
            "if you have determined that there are more steps required to "
            "fully answer the user's original request. ensure that no "
            "further action is taken unless explicitly requested. If you "
            "have selected 'end' as the next_worker then you do not need to "
            "provide instructions."
        )
    )