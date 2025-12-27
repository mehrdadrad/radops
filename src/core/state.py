import logging
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator

from config.config import settings

logger = logging.getLogger(__name__)

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


class WorkerAgentOutput(BaseModel):
    success: bool = Field(
        description="True if the task was completed successfully, False otherwise."
    )
    result: str = Field(
        description="The result of the work or the error message."
    )
    failure_reason: str | None = Field(
        default=None,
        description="If failed, the reason why."
    )

class SupervisorAgentOutput(BaseModel):
    detected_requirements: list[str] = Field(
        description=(
            "A list of specific requirements extracted from the user's request."
        )
    )
    remaining_steps: list[str] = Field(
        description=(
            "The list of detected_requirements that are NOT yet present in "
            "completed_steps OR failed_steps. "
            "If this list is not empty, 'is_fully_completed' MUST be False."
        )
    )
    completed_steps: list[str] = Field(
        description=(
            "A list of steps that have been successfully completed so far. "
            "Include steps where the tool ran successfully but returned no results (e.g., 'not found')."
        )
    )
    failed_steps: list[str] = Field(
        default_factory=list,
        description=(
            "List of steps that were ATTEMPTED but FAILED (e.g., 'Tool not found', "
            "'Permission denied', 'Timeouts'). Tracking this prevents infinite retries."
        )
    )
    is_fully_completed: bool = Field(
        description=(
            "Set to True if EVERY item in 'detected_requirements' has been addressed, "
            "either by being listed in 'completed_steps' (Success) "
            "OR by being listed in 'failed_steps' (Failure/Give Up). "
            "If we have tried and failed, the job is technically COMPLETED."
        )
    )
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
            "4. **Final Step:** provide a verification based on the 'is_fully_completed' is false or true at the end of the conversation."
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

    @model_validator(mode='after')
    def prevent_insanity_loop(self):
        if self.next_worker != 'end':
            for failure in self.failed_steps:
                if self.next_worker.replace("_", " ") in failure.lower():
                    raise ValueError(
                        f"STOP: You are assigning '{self.next_worker}' to a task, "
                        f"but they already failed: '{failure}'. "
                        "Do not retry. Mark 'is_fully_completed' = True (because we gave up) and route to 'end'."
                    )
        if self.next_worker == 'end':
            total_attempted = len(self.completed_steps) + len(self.failed_steps)
            if total_attempted < len(self.detected_requirements):
                pass

        return self

    @model_validator(mode='after')
    def force_completion_flag(self):
        all_attempts = self.completed_steps + self.failed_steps
        
        if len(all_attempts) >= len(self.detected_requirements):
            
            if not self.is_fully_completed:
                logger.info("Forced supervisor completion flag.")
                self.is_fully_completed = True
                
                # Optional: Force the next step to END if we are "done"
                if self.next_worker != 'end':
                    self.next_worker = WorkerEnum.end

                    
        return self
