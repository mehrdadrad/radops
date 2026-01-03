from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.telemetry.telemetry import Telemetry

def register_default_metrics(telemetry: "Telemetry"):
    """Registers default metrics for the application."""
    telemetry.register_counter(
        "agent.invocations.total",
        description="Total number of times the agent node is invoked."
    )
    telemetry.register_histogram(
        "agent.llm.duration_seconds",
        unit="s",
        description="Duration of the LLM call in the agent node."
    )
    telemetry.register_counter(
        "agent.llm.tokens.total",
        description="Total number of tokens used by LLM calls."
    )
    telemetry.register_histogram(
        "agent.tool.duration_seconds",
        unit="s",
        description="Duration of tool execution."
    )
    telemetry.register_counter(
        "agent.tool.invocations.total",
        description="Total number of tool executions."
    )
    telemetry.register_histogram(
        "agent.auditor.score",
        description="Quality assurance score assigned by the auditor."
    )
    telemetry.register_counter(
        "agent.llm.errors",
        description="Total number of LLM errors."
    )
    telemetry.register_counter(
        "agent.tool.errors",
        description="Total number of tool execution errors."
    )
