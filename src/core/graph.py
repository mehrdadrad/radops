"""This module defines and configures the main LangGraph for the agent."""
import logging
import time
from typing import Any, AsyncGenerator, Literal, Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from config.config import settings
from core.auth import is_tool_authorized
from core.llm import llm_factory
from core.memory import get_mem0_client
from core.state import State, SupervisorAgentOutput
from prompts.system import EXTENSION_PROMPT, SUPERVISOR_PROMPT, SYSTEM_PROMPT
from services.guardrails.guardrails import guardrail
from services.telemetry.telemetry import Telemetry
from tools import ToolRegistry

logger = logging.getLogger(__name__)

telemetry = Telemetry()


async def run_graph(checkpointer=None, tools=None, tool_registry=None):
    """Builds and runs the LangGraph application."""
    if tool_registry is None:
        tool_registry = ToolRegistry(checkpointer=checkpointer)
    if tools is None:
        tools = await tool_registry.get_all_tools()

    graph_builder = StateGraph(State)
    graph_builder.add_node("guardrail", guardrail)
    graph_builder.add_node("memory", manage_memory_node)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node(
        "common_agent", create_agent("common", SYSTEM_PROMPT, tools)
    )
    graph_builder.add_node(
        "tools",
        ToolNode(
            tools=tools,
            awrap_tool_call=authorize_tools,
            handle_tool_errors=custom_error_handler,
        ),
        retry_policy=RetryPolicy(max_attempts=2),
    )
    graph_builder.add_conditional_edges(
        "supervisor",
        route_workflow,
    )
    graph_builder.add_conditional_edges(
        "common_agent",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor", "end": END},
    )
    graph_builder.add_conditional_edges(
        "tools",
        route_back_from_tool,
    )
    graph_builder.add_edge("memory", "supervisor")
    graph_builder.add_edge(START, "guardrail")
    graph_builder.add_conditional_edges(
        "guardrail", check_end_status, {"end": END, "continue": "memory"}
    )

    # setup configured agent(s)
    for agent_name, agent_config in settings.agent.profiles.items():  # pylint: disable=no-member
        system_prompt_file = agent_config.system_prompt_file
        if system_prompt_file:
            with open(system_prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        else:
            raise ValueError("System prompt file not found")
        logger.info("Configuring agent: %s", agent_name)
        graph_builder.add_node(
            agent_name,
            create_agent(
                agent_name, system_prompt, tools, agent_config.llm_profile
            ),
        )
        graph_builder.add_conditional_edges(
            agent_name,
            route_after_worker,
            {"tools": "tools", "supervisor": "supervisor", "end": END},
        )

    graph = graph_builder.compile(checkpointer=checkpointer)

    if logger.level == logging.DEBUG:
        try:
            graph.get_graph().draw_mermaid_png(
                output_file_path="graph.png", max_retries=5, retry_delay=2.0
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to draw graph: %s", e)

    return graph


def create_agent(
    agent_name: str,
    system_prompt: str,
    tools: Sequence[BaseTool],
    llm_profile: str = None,
):
    """Factory function to create a worker agent node."""
    effective_llm_profile = (
        llm_profile
        if llm_profile
        else settings.llm.default_profile  # pylint: disable=no-member
    )

    def agent(state: State):
        telemetry.update_counter(
            "agent.invocations.total", attributes={"agent": agent_name}
        )
        llm = llm_factory(effective_llm_profile)
        llm_with_tools = llm.bind_tools(tools)

        user_id = state.get("user_id", "User")
        prompt = f"{system_prompt}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
        messages = construct_llm_context(state, prompt)
        start_time = time.perf_counter()
        ai_response = llm_with_tools.invoke(messages)
        duration = time.perf_counter() - start_time
        telemetry.update_histogram(
            "agent.llm.duration_seconds",
            duration,
            attributes={"agent": agent_name},
        )

        nodes = state["response_metadata"].get("nodes", []) + [agent_name]
        return {"messages": [ai_response], "response_metadata": {"nodes": nodes}}

    return agent


def supervisor_node(state: State) -> dict:
    """The supervisor node that routes to the correct worker or ends."""
    node_name = "supervisor"
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": node_name}
    )

    llm = llm_factory(settings.llm.default_profile)  # pylint: disable=no-member
    llm_structured = llm.with_structured_output(SupervisorAgentOutput)

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]

    start_time = time.perf_counter()
    decision = llm_structured.invoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": node_name}
    )

    nodes = state["response_metadata"].get("nodes", []) + [node_name]
    return {
        "next_worker": decision.next_worker.value,
        "response_metadata": {"nodes": nodes},
    }


async def manage_memory_node(state: State) -> dict:
    """Manages short-term and long-term memory for the conversation."""
    logger.info("Mem0: Managing memory...")
    mem0 = await get_mem0_client()
    messages = state["messages"]
    user_id = state["user_id"]

    # search
    try:
        memories = await mem0.search(messages[-1].content, user_id=user_id)
        memory_list = memories["results"]
        context = "Relevant information from previous conversations:\n"
        for memory in memory_list:
            context += f"- {memory['memory']}\n"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("mem0 err: %s", e)
        context = ""

    # Add to memory if we have a request-response pair
    human_messages = [
        msg for msg in state["messages"] if isinstance(msg, HumanMessage)
    ]
    last_human_message = (
        human_messages[-2]
        if len(human_messages) > 1
        else human_messages[-1]
        if human_messages
        else None
    )
    last_ai_message = next(
        (
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ),
        None,
    )
    if last_human_message and last_ai_message:
        interaction_messages = [
            {"role": "user", "content": last_human_message.content},
            {"role": "assistant", "content": last_ai_message.content},
        ]
        await mem0.add(messages=interaction_messages, user_id=user_id)
        logger.info("Mem0: Added interaction for user '%s'.", user_id)

    # Remove old messages
    delete_messages = []
    messages_to_keep = (
        settings.memory.summarization.keep_message  # pylint: disable=no-member
    )
    if messages_to_keep:
        delete_messages = [
            RemoveMessage(id=m.id) for m in state["messages"][:-messages_to_keep]
        ]

    return {"relevant_memories": context, "messages": delete_messages}


async def astream_graph_updates(
    graph: StateGraph, user_input: str, user_id: str
) -> AsyncGenerator[AIMessage, Any]:
    """Streams updates from the graph execution."""
    config = {"configurable": {"thread_id": user_id}}
    initial_input = {
        "messages": [{"role": "user", "content": user_input}],
        "user_id": user_id,
        "relevant_memories": "",  # Start with no memories, they will be fetched
        "next_worker": "",
        "response_metadata": {},
    }
    async for event in graph.astream(
        initial_input,
        config,
        stream_mode="values",
    ):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage):
            yield last_message, event["response_metadata"]


async def authorize_tools(request, handler) -> ToolMessage:
    """Checks if a tool is authorized for the user before execution."""
    tool_name = request.tool_call["name"]
    user_id = request.state.get("user_id")

    if is_tool_authorized(tool_name, user_id):
        return await handler(request)

    error_message = f"unauthorized tool call: {tool_name}"
    return ToolMessage(
        content=error_message,
        name=tool_name,
        tool_call_id=request.tool_call["id"],
        status="error",
    )


def route_workflow(state: State) -> str:
    """Routes to the next worker based on the supervisor's decision."""
    return state["next_worker"]


def route_back_from_tool(state: State) -> str:
    """Routes back to the designated worker after a tool call."""
    return state.get("next_worker")


def route_after_worker(state: State) -> Literal["tools", "supervisor", "end"]:
    """
    Determines the next step after a worker agent has run.

    - If tool calls are present, route to the 'tools' node to execute them.
    - If the agent signals to escalate, route back to the 'supervisor'.
    - Otherwise, end the workflow.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "tools"
        # Instruct the agent to include the word "supervisor" in its response
        # to trigger escalation.
        if "supervisor" in last_message.content.lower():
            return "supervisor"
    return "end"


def check_end_status(state: State) -> Literal["end", "continue"]:
    """Checks if the guardrail has marked the conversation to end."""
    if state["end_status"] == "end":
        return "end"
    return "continue"


def tools_condition(state: State) -> str:
    """Decides whether to call tools or end the process."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "end"


def custom_error_handler(e: Exception) -> str:
    """Handles errors during tool execution."""
    logger.error("critical failure: %s", e)

    return f"The tool failed to execute. error details: {str(e)}."


def construct_llm_context(state: State, system_prompt: str):
    """Constructs the full context for the LLM from state."""
    relevant_memories = state.get("relevant_memories")
    memory_message = []
    if relevant_memories:
        memory_message = [
            SystemMessage(
                content="Here is relevant information from your memory:\n"
                f"{relevant_memories}"
            )
        ]

    messages = (
        [SystemMessage(content=system_prompt)]
        + memory_message
        + state["messages"]
    )

    return messages
