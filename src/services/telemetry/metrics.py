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
