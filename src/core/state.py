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
members["human"] = "human"
members["end"] = "end"
WorkerEnum = Enum("WorkerEnum", members, type=str)

class Requirement(BaseModel):
    """Represents a single requirement with an assigned agent."""
    id: int = Field(description="The unique identifier for this step, starting at 1.")
    instruction: str = Field(description="The specific requirement or step.")
    assigned_agent: WorkerEnum = Field(description="The agent that MUST handle this step. Use 'human' for approvals.")

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
    steps_status: list[Literal["pending", "completed", "failed", "skipped"]]


class WorkerAgentOutput(BaseModel):
    """Output model for worker agents."""
    success: bool = Field(
        description="True if the tool executed correctly (even if no data was found). "
        "False ONLY if a technical error occurred."
    )
    failure_reason: str | None = Field(
        default=None,
        description="If failed, the reason why."
    )

class SupervisorAgentOutputBase(BaseModel):
    """Output model for the supervisor agent."""
    current_step_id: int = Field(
        description=(
            "The ID of the current step"
        )
    )
    current_step_status: Literal["pending", "completed", "failed", "skipped"] = Field(
        description=(
            "The status of the current step. "
        )
    )
    skipped_step_ids: list[int] = Field(
        default=[],
        description="List of FUTURE step IDs to mark as skipped immediately."
    )
    next_worker: WorkerEnum = Field(
        description=(
            "The single worker responsible for the very next action. "
            "Only select 'end' if there is no more pending steps. "
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
            "You do not need to repeat findings from 3 steps ago "
            "(unless relevant to the current context)."
            "3. **NO AGGREGATION:** Do NOT provide a final summary of all steps at the end. "
            "Only report the result of the *current* step. "
            "The user has already seen the previous steps."
            "4. **Briefness:** If the result is huge (e.g. long logs), summarize it. "
            "Do not hit the token limit."
            "5. **Errors:** If a tool failed, timed out, or is partially available, "
            "you MUST report the specific error message."
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
            "CRITICAL RULES:\n"
            "1. **Decompose**: Split complex requests into individual steps.\n"
            "2. **Approvals**: If the request implies approval (e.g. 'if I approve', 'ask me first'), "
            "you MUST create a dedicated step for the 'human' agent to get confirmation. "
            "This 'human' step must be BEFORE the action that requires approval.\n"
            "3. **Completeness**: Include ALL steps (both completed and pending). "
            "Do NOT remove steps that have been finished."
        )
    )
    response_to_user: str = Field(
        description=(
            "Communication to the user. Since this is the planning phase, you MUST:\n"
            "1. Acknowledge the user's request.\n"
            "2. Present the plan you have generated (the steps in `detected_requirements`) to the user so they know what will happen.\n"
            "3. Inform them you are starting the first step."
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
    