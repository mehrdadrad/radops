"""Guardrails service to check user input safety."""
import time
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from config.config import settings
from core.state import State
from core.llm import llm_factory
from services.telemetry.telemetry import telemetry


class GuardrailsOutput(BaseModel):
    """Output model for the safety and relevance guardrails agent."""
    is_safe: bool = Field(
        description=(
            "True if the input is safe and relevant, "
            "False if it is a jailbreak attempt or irrelevant."
        )
    )
    reasoning: str = Field(description="Brief explanation of the safety decision.")


def guardrails(state: State):
    """Check if the user input is safe and relevant."""
    node_name = "guardrails"
    if not settings.agent.guardrails.enabled:
        return {
            "messages": [],
            "summary": state.get("summary", None),
            "end_status": "continue"
        }

    telemetry.update_counter("agent.invocations.total", attributes={"agent": node_name})

    llm = llm_factory(settings.agent.guardrails.llm_profile)
    llm_structured = llm.with_structured_output(GuardrailsOutput)

    user_input = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]

    with open(settings.agent.guardrails.prompt_file, "r", encoding="utf-8") as f:
        guardrail_instructions = f.read()

    content = user_input[-1].content if user_input else ''
    prompt = f"{guardrail_instructions}\n\nUser Input: {content}"
    start_time = time.perf_counter()
    guardrails_result = llm_structured.invoke(prompt)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": node_name}
    )

    if not guardrails_result.is_safe:
        telemetry.update_counter("guardrails.blocked.total")
        error_message = (
            f"🚨 I cannot assist with that request. Reason: {guardrails_result.reasoning}"
        )
        return {
            "messages": [AIMessage(content=error_message)],
            "response_metadata": {"nodes": [node_name]},
            "end_status": "end",
        }

    return {
        "messages": [],
        "response_metadata": {"nodes": [node_name]},
        "end_status": "continue"
    }
