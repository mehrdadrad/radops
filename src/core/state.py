"""Module for defining the state and output models for the agent."""
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

class Requirement(BaseModel):
    """Represents a single requirement with an assigned agent."""
    instruction: str = Field(description="The specific requirement or step.")
    assigned_agent: WorkerEnum = Field(description="The agent that MUST handle this step.")

class State(TypedDict):
    """Represents the state of the agent execution graph."""
    messages: Annotated[list, add_messages]
    summary: str
    user_id: str
    task_id: str
    end_status: Literal["end", "continue"]
    relevant_memories: str
    next_worker: str
    response_metadata: dict[str, Any]
    detected_requirements: list[Requirement]


class WorkerAgentOutput(BaseModel):
    """Output model for worker agents."""
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

class SupervisorAgentOutputBase(BaseModel):
    """Output model for the supervisor agent."""
    completed_steps: list[str] = Field(
        description=(
            "A list of steps that have been successfully completed so far. "
            "Include steps where the tool ran successfully but returned no "
            "results (e.g., 'not found')."
        )
    )
    failed_steps: list[str] = Field(
        default_factory=list,
        description=(
            "List of steps that were ATTEMPTED but FAILED, or SKIPPED due to conditions. "
            "Tracking this prevents infinite retries."
        )
    )
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
            "Communication to the user. You must act as a 'Live Reporter'. "
            "Follow these strict rules for reporting data:"
            
            "1. **Report Immediately:** As soon as you receive data from a worker"
            "you MUST include it in this response. Do not hold it back for a 'final summary'."
            
            "2. **Incremental Updates Only:** Focus on what *just* happened in the last step. "
            "You do not need to repeat findings from 3 steps ago (unless relevant to the current context)."
            
            "3. **NO AGGREGATION:** Do NOT provide a final summary of all steps at the end. Only report the result of the *current* step. The user has already seen the previous steps."
            
            "4. **Briefness:** If the result is huge (e.g. long logs), summarize it. Do not hit the token limit."
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
        """Prevents the supervisor from routing back to a failed worker."""
        next_worker_str = self.next_worker.value if hasattr(self.next_worker, "value") else str(self.next_worker)
        if next_worker_str != 'end':
            for failure in self.failed_steps:
                if next_worker_str.replace("_", " ").lower() in failure.lower():
                    raise ValueError(
                        f"STOP: You are assigning '{next_worker_str}' to a task, "
                        f"but they already failed: '{failure}'. "
                        "Do not retry. Route to 'end'."
                    )

        return self

    @model_validator(mode='after')
    def validate_worker_instructions(self):
        """Validates that instructions for the worker are descriptive enough."""
        if self.next_worker != 'end' and len(self.instructions_for_worker.strip()) < 5:
            raise ValueError(
                "Instructions for the worker must be descriptive."
            )
        return self

class SupervisorAgentPlanOutput(SupervisorAgentOutputBase):
    """Output model for the supervisor agent (first round)."""
    detected_requirements: list[Requirement] = Field(
        description=(
            "The COMPLETE list of requirements extracted from the user's ORIGINAL request. "
            "CRITICAL: If the user provides a numbered list (e.g. '1- do X, 2- do Y'), you MUST generate a separate Requirement for EACH item. "
            "You MUST include ALL steps (both completed and pending) in this list. "
            "Do NOT remove steps that have been finished."
        )
    )
   
class SupervisorAgentOutput(SupervisorAgentOutputBase):
    """Output model for the supervisor agent (non-first round)."""
    pass

class VerificationResult(BaseModel):
    """Result of a single verification step by the auditor."""
    is_success: bool = Field(
        description=(
            "True ONLY if the 'actual_evidence' fully satisfies the 'original_requirement'."
        )
    )
    missing_information: str = Field(
        description=(
            "If is_success is False, describe EXACTLY what is missing or incorrect. "
            "If True, return an empty string."
        )
    )
    correction_instruction: str = Field(
        description=(
            "Precise instruction for the Supervisor to fix the error. "
            "If True, return an empty string."
        )
    )


class AuditReport(BaseModel):
    """Report generated by the auditor agent."""
    # We grade every single requirement individually
    verifications: list[VerificationResult]
    score: float = Field(
        description=(
            "A score between 0.0 and 1.0 indicating the quality and completeness of the execution. "
            "1.0 is perfect."
        )
    )

    final_decision: Literal["APPROVE", "REJECT"] = Field(
        description="APPROVE only if ALL verifications are True."
    )
    