"""This module defines and configures the main LangGraph for the agent."""
import logging
import re
import time
from functools import partial
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
from core.state import State, SupervisorAgentOutput, AuditReport
from prompts.system import EXTENSION_PROMPT, SUPERVISOR_PROMPT, PLATFORM_PROMPT, AUDITOR_PROMPT
from services.guardrails.guardrails import guardrail
from services.telemetry.telemetry import telemetry
from tools import ToolRegistry

logger = logging.getLogger(__name__)


async def run_graph(checkpointer=None, tools=None, tool_registry=None):
    """Builds and runs the LangGraph application."""
    if tool_registry is None:
        tool_registry = ToolRegistry(checkpointer=checkpointer)
    if tools is None:
        tools = await tool_registry.get_all_tools()

    system_tools = await tool_registry.get_system_tools()

    graph_builder = StateGraph(State)
    graph_builder.add_node("guardrail", guardrail)
    graph_builder.add_node("memory", manage_memory_node)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("auditor", auditor_node)
    graph_builder.add_node("system", partial(system_node, tools=system_tools))
    graph_builder.add_node(
        "tools",
        ToolNode(
            tools=tools + system_tools,
            awrap_tool_call=authorize_tools,
            handle_tool_errors=custom_error_handler,
        ),
        retry_policy=RetryPolicy(max_attempts=2),
    )

    # Create path mapping
    path_map = {name: name for name in settings.agent.profiles.keys()}
    path_map["tools"] = "tools"
    path_map["supervisor"] = "supervisor"
    path_map["system"] = "system"
    path_map["auditor"] = "auditor"
    path_map["end"] = END

    graph_builder.add_conditional_edges(
        "supervisor",
        route_workflow,
        path_map,
    )
    graph_builder.add_conditional_edges(
        "auditor",
        route_auditor,
        {"approved": END, "supervisor": "supervisor"},
    )
    graph_builder.add_conditional_edges(
        "tools",
        route_back_from_tool,
        path_map,
    )
    graph_builder.add_edge("memory", "supervisor")
    graph_builder.add_edge(START, "guardrail")
    graph_builder.add_conditional_edges(
        "guardrail",
        check_end_status,
        {"end": END, "continue": "memory"},
    )
    graph_builder.add_conditional_edges(
        "system",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor"},
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
        filtered_tools = filter_tools(tools, agent_config.allow_tools)
        graph_builder.add_node(
            agent_name,
            create_agent(
                agent_name, system_prompt, filtered_tools, agent_config.llm_profile
            ),
        )
        graph_builder.add_conditional_edges(
            agent_name,
            route_after_worker,
            {"tools": "tools", "supervisor": "supervisor"},
        )
        logger.info("Agent configured: %s with %d tools.", agent_name, len(filtered_tools))


    graph = graph_builder.compile(checkpointer=checkpointer)

    if logger.level != logging.DEBUG:
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

    llm_profile = (
        settings.agent.supervisor.llm_profile or settings.llm.default_profile  # pylint: disable=no-member
    )
    logger.info("Supervisor LLM profile: %s", llm_profile)
    llm = llm_factory(llm_profile)
    llm_structured = llm.with_structured_output(SupervisorAgentOutput)

    # Sanitize messages to remove orphaned ToolMessages caused by truncation
    conversation_messages = state["messages"]
    while conversation_messages and isinstance(conversation_messages[0], ToolMessage):
        conversation_messages = conversation_messages[1:]

    conversation_messages = sanitize_tool_calls(conversation_messages)

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + conversation_messages

    start_time = time.perf_counter()
    decision = llm_structured.invoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": node_name}
    )

    nodes = state["response_metadata"].get("nodes", []) + [node_name]
    ai_message = AIMessage(
        content=decision.response_to_user
    )

    if decision.next_worker == "end":
        logger.info("requirements: %s", decision.detected_requirements)
        logger.info("completed steps: %s", decision.completed_steps)
        logger.info("failed steps: %s", decision.failed_steps)
        logger.info("is_fully_completed: %s", decision.is_fully_completed)
        return {
            "next_worker": "end",
            "messages": [ai_message],
            "response_metadata": {"nodes": nodes},
        }

    context_message = HumanMessage(
        content=(
            "COMMAND FROM SUPERVISOR: "
            f"{decision.instructions_for_worker}, "
            "When finished or if you cannot proceed, use the 'system__submit_work' tool "
            "to report the result."
        ),
        name="supervisor"
    )

    return {
        "next_worker": decision.next_worker.value,
        "response_metadata": {"nodes": nodes},
        "messages": [context_message, ai_message],
    }

def system_node(state: State, tools: Sequence[BaseTool]) -> dict:
    """The system node that manages the infrastructure of the bot itself."""
    node_name = "system"
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": node_name}
    )

    llm_profile = (
        settings.agent.system.llm_profile or settings.llm.default_profile  # pylint: disable=no-member
    )
    llm = llm_factory(llm_profile)
    llm_with_tools = llm.bind_tools(tools)

    user_id = state.get("user_id", "User")
    prompt = f"{PLATFORM_PROMPT}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
    messages = construct_llm_context(state, prompt)

    start_time = time.perf_counter()
    decision = llm_with_tools.invoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": node_name}
    )

    nodes = state["response_metadata"].get("nodes", []) + [node_name]
    return {"messages": [decision], "response_metadata": {"nodes": nodes}}

def auditor_node(state):
    if not settings.agent.auditor.enabled:
        return {}

    agent_name = "auditor"
    messages = state["messages"]
    original_request = "Unknown Request"
    last_request_index = -1

    llm_profile = (
        settings.agent.auditor.llm_profile or settings.llm.default_profile  # pylint: disable=no-member
    )

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) != "supervisor":
            original_request = msg.content
            last_request_index = i
            break
    tool_outputs = [
        m for i, m in enumerate(messages)
        if isinstance(m, ToolMessage) and i > last_request_index
    ]
    supervisor_response = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    relevant_memories = state.get("relevant_memories", "")

    llm = llm_factory(llm_profile)
    llm_structured = llm.with_structured_output(AuditReport)
    audit = llm_structured.invoke([
        SystemMessage(content=AUDITOR_PROMPT),
        HumanMessage(content=f"User Request: {original_request}"),
        HumanMessage(content=f"Memory Evidence (Past Knowledge): {relevant_memories}"),
        HumanMessage(content=f"Tool Evidence: {tool_outputs}"),
        HumanMessage(content=f"Supervisor Conclusion: {supervisor_response}")
    ])

    nodes = state["response_metadata"].get("nodes", []) + [agent_name]

    if audit.score >= settings.agent.auditor.threshold:
        logger.info(
            "QA Score %s is above threshold %s. Finishing job.",
            audit.score,
            settings.agent.auditor.threshold,
        )
        return {
            "messages": [
                AIMessage(content=f"QA Passed. Finishing job (score: {audit.score})")
            ],
            "response_metadata": {"nodes": nodes},
        }
    
    failed_verifications = [v for v in audit.verifications if not v.is_success]

    if failed_verifications:
        # Check for previous rejections to prevent infinite loops
        previous_rejections = [
            m for m in messages
            if isinstance(m, HumanMessage) and "QA REJECTION" in str(m.content)
        ]
        if len(previous_rejections) >= 2:
            logger.info(
                f"QA rejected {len(previous_rejections)} times (Last score {audit.score}) "
                "Ending task."
            )
            return {
                "response_metadata": {"nodes": nodes}
            }

        first_failure = failed_verifications[0]
        feedback = f"QA REJECTION: {first_failure.missing_information}. {first_failure.correction_instruction}"
        logger.info(
            f"QA Rejected with feedback (Score {audit.score}): {feedback}"
        )
        return {
            "messages": [HumanMessage(content=feedback)],
            "next_worker": None,
            "response_metadata": {"nodes": nodes}
        }

    logger.info("QA Passed with score %s. Finishing job.", audit.score)
    return {
        "messages": [
            AIMessage(content=f"QA Passed. Finishing job (score: {audit.score})")
        ],
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
        memories = await mem0.search(
            messages[-1].content, user_id=user_id, limit=settings.mem0.limit
        )
        memory_list = memories["results"]
        context = "Relevant information from previous conversations:\n"
        for memory in memory_list:
            context += f"- {memory['memory']}\n"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("mem0 err: %s", e)
        context = ""

    # Add to memory if we have a request-response pair
    human_messages = [
        msg for msg in state["messages"]
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) != "supervisor"
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
        await mem0.add(
            messages=interaction_messages, 
            user_id=user_id,
        )
        logger.info("Mem0: Added interaction for user '%s'.", user_id)

    # Remove old messages
    delete_messages = []
    messages_to_keep = (
        settings.memory.summarization.keep_message  # pylint: disable=no-member
    )
    if messages_to_keep:
        delete_messages = [
            RemoveMessage(id=m.id)
            for m in state["messages"][:-messages_to_keep]
        ]

    return {"relevant_memories": context, "messages": delete_messages}


async def astream_graph_updates(
    graph: StateGraph, user_input: str, user_id: str
) -> AsyncGenerator[AIMessage, Any]:
    """Streams updates from the graph execution."""
    config = {
        "configurable": {"thread_id": user_id},
        "recursion_limit": settings.graph.recursion_limit,
        "max_concurrency": settings.graph.max_concurrency,
    }
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
    next_worker = state["next_worker"]
    if next_worker == "end":
        return "auditor"
    return next_worker

def route_auditor(state):
    last_msg = state["messages"][-1]
    if "QA REJECTION" in last_msg.content:
        return "supervisor"
    else:
        return "approved"


def route_back_from_tool(state: State) -> str:
    """Routes back to the designated worker after a tool call."""
    # Check if the escalation tool was called
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            if message.name == "system__submit_work":
                return "supervisor"
        else:
            break
    return state.get("next_worker")


def route_after_worker(state: State) -> Literal["tools", "supervisor"]:
    """
    Determines the next step after a worker agent has run.

    - If tool calls are present, route to the 'tools' node to execute them.
    - Otherwise, end the workflow.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            if detect_tool_loop(state):
                logger.warning("Tool loop detected. Ending worker execution.")
                return "supervisor"
            return "tools"
    return "supervisor"


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

    # Sanitize messages to remove orphaned ToolMessages caused by truncation
    conversation_messages = state["messages"]
    while conversation_messages and isinstance(conversation_messages[0], ToolMessage):
        conversation_messages = conversation_messages[1:]

    conversation_messages = sanitize_tool_calls(conversation_messages)

    messages = (
        [SystemMessage(content=system_prompt)]
        + memory_message
        + conversation_messages
    )

    return messages


def filter_tools(
    tools: Sequence[BaseTool], allow_list: Sequence[str] = None
) -> Sequence[BaseTool]:
    """Filters tools based on name or regex pattern."""
    if allow_list is None:
        return tools
    if len(allow_list) == 0:
        return []

    filtered_tools = []
    for tool in tools:
        for pattern in allow_list:
            if re.fullmatch(pattern, tool.name):
                filtered_tools.append(tool)
                break
    return filtered_tools

def sanitize_tool_calls(messages: list) -> list:
    """
    Sanitizes messages to ensure AIMessages with tool_calls are followed by ToolMessages.
    If not, the tool_calls are removed to prevent API errors (400).
    """
    sanitized = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            is_followed_by_tool = False
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if isinstance(next_msg, ToolMessage):
                    is_followed_by_tool = True
            
            if not is_followed_by_tool:
                content = msg.content if msg.content else "..."
                sanitized.append(AIMessage(content=content, id=msg.id))
            else:
                sanitized.append(msg)
        else:
            sanitized.append(msg)
    return sanitized

def detect_tool_loop(state, limit=3):
    messages = state['messages']

    # Find the start of the current conversation turn
    last_human_idx = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) != "supervisor":
            last_human_idx = i
            break

    # Filter only AI messages that have tool calls
    tool_calls = [
        m.tool_calls[0]
        for m in messages[last_human_idx:]
        if isinstance(m, AIMessage) and m.tool_calls
    ]

    # Check if we have enough history to detect a loop
    if len(tool_calls) < limit:
        return False

    # Check for identical calls (Name AND Args)
    last_call = tool_calls[-1]
    if all(
        (t['name'] == last_call['name'] and t['args'] == last_call['args'])
        for t in tool_calls[-limit:]
    ):
        return True

    # Check for alternating loops (e.g., A, B, A, B)
    if len(tool_calls) >= 4:
        names = [t['name'] for t in tool_calls[-4:]]
        if names[0] == names[2] and names[1] == names[3]:
            return True

    return False