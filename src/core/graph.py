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
from prompts.system import (
    CLOUD_SPECIALIST_PROMPT,
    EXTENSION_PROMPT,
    NETWORK_SPECIALIST_PROMPT,
    SUPERVISOR_PROMPT,
    SYSTEM_PROMPT,
)
from services.guardrails.guardrails import guardrail
from services.telemetry.telemetry import Telemetry
from tools import ToolRegistry

logger = logging.getLogger(__name__)

telemetry = Telemetry()

async def run_graph(checkpointer=None, tools=None):
    """Builds and runs the LangGraph application."""
    tool_registry = ToolRegistry(checkpointer=checkpointer)
    if tools is None:
        tools = await tool_registry.get_all_tools()

    graph_builder = StateGraph(State)
    graph_builder.add_node("guardrail", guardrail)
    graph_builder.add_node("memory", manage_memory_node)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("common_agent", lambda state: agent_node(state, tools))
    graph_builder.add_node("react_agent", lambda state: react_node(state, tools))
    graph_builder.add_node("cloud_agent", lambda state: cloud_node(state, tools))
    graph_builder.add_node(
        "tools",
        ToolNode(
            tools=tools,
            awrap_tool_call=authorize_tools,
            handle_tool_errors=custom_error_handler
        ),
        retry_policy=RetryPolicy(max_attempts=2)
    )
    graph_builder.add_conditional_edges(
        "supervisor",
        route_workflow,
        {
            "common_agent": "common_agent",
            "react_agent": "react_agent",
            "cloud_agent": "cloud_agent",
        },
    )
    graph_builder.add_conditional_edges(
        "common_agent",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor", "end": END},
    )
    graph_builder.add_conditional_edges(
        "react_agent",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor", "end": END},
    )
    graph_builder.add_conditional_edges(
        "cloud_agent",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor", "end": END},
    )
    graph_builder.add_conditional_edges(
        "tools",
        route_back_from_tool,
        {
            "common_agent": "common_agent",
            "react_agent": "react_agent",
            "cloud_agent": "cloud_agent",
        }
    )
    graph_builder.add_edge("memory", "supervisor")
    graph_builder.add_edge(START, "guardrail")
    graph_builder.add_conditional_edges(
        "guardrail",
        check_end_status,
        {"end": END, "continue": "memory"}
    )
    graph = graph_builder.compile(checkpointer=checkpointer)

    if logger.level == logging.DEBUG:
        try:
            graph.get_graph().draw_mermaid_png(
                output_file_path="graph.png",
                max_retries=5,
                retry_delay=2.0
            )
        except Exception as e:
            logger.warning(f"Failed to draw graph: {e}")

    return graph

def agent_node(state: State, tools: Sequence[BaseTool]):
    """Invokes the LLM with the current state and available tools."""
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": "common"}
    )

    llm = llm_factory(settings.llm.default_profile)
    llm_with_tools = llm.bind_tools(tools)

    user_id = state.get("user_id", "User")
    system_prompt = (
        f"{SYSTEM_PROMPT}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
    )

    relevant_memories = state.get("relevant_memories")
    memory_message = []
    if relevant_memories:
        memory_message = [SystemMessage(
            content=f"Here is relevant information from your memory:\n"
                    f"{relevant_memories}"
        )]

    messages = (
        [SystemMessage(content=system_prompt)] +
        memory_message +
        state["messages"]
    )

    start_time = time.perf_counter()
    ai_response = llm_with_tools.invoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": "common"}
    )

    return {"messages": [ai_response]}

def supervisor_node(state: State):
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": "supervisor"}
    )

    llm = llm_factory(settings.llm.default_profile)
    llm_structured = llm.with_structured_output(SupervisorAgentOutput)

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]

    decision = llm_structured.invoke(messages)
    return {"next_worker": decision.next_worker}

def react_node(state: State, tools: Sequence[BaseTool]):
    llm = llm_factory(settings.llm.default_profile)
    llm_with_tools = llm.bind_tools(tools)

    user_id = state.get("user_id", "User")
    system_prompt = (
        f"{NETWORK_SPECIALIST_PROMPT}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    start_time = time.perf_counter()
    response = llm_with_tools.invoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": "supervisor"}
    )

    return {"messages": [response]}

def cloud_node(state: State, tools: Sequence[BaseTool]):
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": "cloud"}
    )
    llm = llm_factory(settings.llm.default_profile)
    llm_with_tools = llm.bind_tools(tools)

    user_id = state.get("user_id", "User")
    system_prompt = (
        f"{CLOUD_SPECIALIST_PROMPT}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    start_time = time.perf_counter()
    response = llm_with_tools.invoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": "cloud"}
    )

    return {"messages": [response]}

async def manage_memory_node(state: State):
    """
    Adds current interaction to mem0 and retrieves relevant memories.
    """
    logging.info("Mem0: Managing memory...")
    mem0 = await get_mem0_client()
    user_id = state["user_id"]

    human_messages = [
        msg for msg in state["messages"] if isinstance(msg, HumanMessage)
    ]
    last_human_message = (
        human_messages[-2] if len(human_messages) > 1
        else human_messages[-1] if human_messages else None
    )
    last_ai_message = next(
        (msg for msg in reversed(state["messages"])
         if isinstance(msg, AIMessage) and not msg.tool_calls),
        None
    )
    last_toolcall_message = (
        state["messages"][-3]
        if len(state["messages"]) > 2 and
        isinstance(state["messages"][-3], ToolMessage)
        else None
    )
    current_human_message = next(
        (msg for msg in reversed(state["messages"])
         if isinstance(msg, HumanMessage)),
        None
    )

    if last_toolcall_message:
        if (last_toolcall_message.name in settings.mem0.excluded_tools or
                "all" in settings.mem0.excluded_tools):
            return {"relevant_memories": "", "messages": []}

    # Add to memory if we have a request-response pair
    if last_human_message and last_ai_message:
        interaction_messages = [
            {"role": "user", "content": last_human_message.content},
            {"role": "assistant", "content": last_ai_message.content},
        ]
        await mem0.add(
            messages=interaction_messages,
            user_id=user_id
        )
        logging.info(f"Mem0: Added interaction for user '{user_id}'.")

    messages_to_keep = settings.memory.summarization.keep_message
    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][:-messages_to_keep]
    ]

    # Search for relevant memories based on the most recent user query
    if last_human_message:
        search_results = await mem0.search(
            query=current_human_message.content,
            user_id=user_id
        )
        relevant_memories = search_results.get("results", [])
        memories_str = (
            "\n".join([f"- {mem['memory']}" for mem in relevant_memories])
            if relevant_memories else ""
        )
        return {"relevant_memories": memories_str, "messages": delete_messages}

    return {"relevant_memories": "", "messages": delete_messages}

async def astream_graph_updates(
    graph: StateGraph, user_input: str, user_id: str
) -> AsyncGenerator[AIMessage, Any]:
    config = {"configurable": {"thread_id": user_id}}
    initial_input = {
        "messages": [{"role": "user", "content": user_input}],
        "user_id": user_id,
        "relevant_memories": "",  # Start with no memories, they will be fetched
        "next_worker": "",
    }
    async for event in graph.astream(
        initial_input,
        config,
        stream_mode="values",
    ):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage):
            yield last_message

async def authorize_tools(request, handler):
    tool_name = request.tool_call["name"]
    user_id = request.state.get("user_id")

    if is_tool_authorized(tool_name, user_id):
        return await handler(request)
    else:
        error_message = f"unauthorized tool call: {tool_name}"
        return ToolMessage(
            content=error_message,
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

def route_workflow(state: State):
    next_worker = state["next_worker"]

    match next_worker:
        case "network_specialist":
            return "react_agent"
        case "cloud_specialist":
            return "cloud_agent"
        case "atomic_tool":
            return "common_agent"
        case "common_agent":
            return "common_agent"
        case _:
            return "end"

def route_back_from_tool(state):
    who_called_me = state.get("next_worker")

    if who_called_me == "common_agent" or who_called_me == "atomic_tool":
        return "common_agent"
    elif who_called_me == "network_specialist":
        return "react_agent"
    elif who_called_me == "cloud_specialist":
        return "cloud_agent"
    else:
        return END

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

def check_end_status(state: State):
    if state['end_status'] == 'end':
        return "end"
    else:
        return "continue"

def tools_condition(state: State) -> str:
    """A function to decide whether to call tools or end the process."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "end"

def custom_error_handler(e: Exception) -> str:
    logger.error(f"critical failure: {e}")

    return f"The tool failed to execute. error details: {str(e)}."