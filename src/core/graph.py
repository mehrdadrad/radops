"""This module defines and configures the main LangGraph for the agent."""
import json
import logging
import re
import time
import uuid
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
from core.state import State, SupervisorAgentOutput, SupervisorAgentPlanOutput, AuditReport
from prompts.system import (
    EXTENSION_PROMPT,
    SUPERVISOR_PROMPT,
    PLATFORM_PROMPT,
    AUDITOR_PROMPT,
    SUMMARIZATION_PROMPT,
)
from services.guardrails.guardrails import guardrails
from services.telemetry.telemetry import telemetry
from tools import ToolRegistry

logger = logging.getLogger(__name__)

QA_REJECTION_PREFIX = "QA REJECTION"
SYSTEM_SUBMIT_WORK = "system__submit_work"


def get_detected_requirements(state: dict) -> list[dict]:
    """
    Retrieves and normalizes detected requirements from the state.
    Handles Pydantic objects, standard dicts, and LangChain serialized dicts.
    """
    raw_reqs = state.get("detected_requirements", [])
    normalized_reqs = []

    for req in raw_reqs:
        if isinstance(req, dict):
            # Check for LangChain serialized format
            if "kwargs" in req and "lc" in req:
                data = req["kwargs"]
            else:
                data = req

            normalized_reqs.append({
                "id": data.get("id"),
                "instruction": data.get("instruction"),
                "assigned_agent": data.get("assigned_agent")
            })
        else:
            # Assume Pydantic object
            agent = req.assigned_agent
            agent_value = agent.value if hasattr(agent, "value") else str(agent)
            normalized_reqs.append({
                "id": req.id,
                "instruction": req.instruction,
                "assigned_agent": agent_value
            })

    return normalized_reqs


async def run_graph(checkpointer=None, tools=None, tool_registry=None):
    """Builds and runs the LangGraph application."""
    if tool_registry is None:
        tool_registry = ToolRegistry(checkpointer=checkpointer)
    if tools is None:
        tools = await tool_registry.get_all_tools()

    system_tools = await tool_registry.get_system_tools()

    graph_builder = StateGraph(State)
    graph_builder.add_node("guardrails", guardrails)
    graph_builder.add_node("memory", manage_memory_node)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("auditor", auditor_node)
    graph_builder.add_node("human", human_node)
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
    path_map.update({
        "tools": "tools",
        "supervisor": "supervisor",
        "system": "system",
        "auditor": "auditor",
        "human": "human",
        "end": END,
    })

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
    graph_builder.add_edge("human", "supervisor")
    graph_builder.add_edge(START, "guardrails")
    graph_builder.add_conditional_edges(
        "guardrails",
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


    graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human"]
        )

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

    async def agent(state: State):
        telemetry.update_counter(
            "agent.invocations.total", attributes={"agent": agent_name}
        )
        llm = llm_factory(effective_llm_profile, agent_name=agent_name)
        llm_with_tools = llm.bind_tools(tools)

        user_id = state.get("user_id", "User")
        prompt = f"{system_prompt}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
        messages = construct_llm_context(state, prompt)
        start_time = time.perf_counter()
        ai_response = await llm_with_tools.ainvoke(messages)
        duration = time.perf_counter() - start_time
        telemetry.update_histogram(
            "agent.llm.duration_seconds",
            duration,
            attributes={"agent": agent_name},
        )

        nodes = state["response_metadata"].get("nodes", []) + [agent_name]
        return {"messages": [ai_response], "response_metadata": {"nodes": nodes}}

    return agent


def update_step_status(decision, steps_status):
    """Updates the status of the current step based on the supervisor's decision."""
    def _set_status(idx, status):
        if idx < 0:
            logger.warning("Invalid step index: %d", idx)
            return
        while len(steps_status) <= idx:
            steps_status.append("pending")
        steps_status[idx] = status

    if decision.current_step_status != "pending" and decision.current_step_id != 0:
        _set_status(decision.current_step_id - 1, decision.current_step_status)

    if decision.skipped_step_ids:
        for step_id in decision.skipped_step_ids:
            if step_id != 0:
                _set_status(step_id - 1, "skipped")

def enforce_plan(decision, existing_requirements, steps_status):
    """Enforces the plan if the supervisor tries to end early."""
    if existing_requirements and decision.next_worker == "end":
        if len(steps_status) < len(existing_requirements):
            next_req_data = existing_requirements[len(steps_status)]

            assigned_agent = next_req_data.get("assigned_agent")
            instruction = next_req_data.get("instruction") or "No instruction provided"

            if not assigned_agent:
                logger.warning(
                    "Cannot enforce plan: Missing assigned agent for step %d",
                      len(steps_status) + 1,
                )
                return

            decision.next_worker = assigned_agent
            decision.instructions_for_worker = f"Proceed with the next step: {instruction}"
            decision.response_to_user += (
                f"\n\n**Plan Enforcement:** Proceeding to next step: {instruction}"
            )
            short_instruction = (
                instruction[:20] + "..." if len(instruction) > 20 else instruction
            )
            agent_name = assigned_agent
            logger.info(
                "enforcing plan: next worker: %s, instruction: %s",
                agent_name,
                short_instruction,
            )

def check_completion(decision, existing_requirements, steps_status):
    """Forces the decision to 'end' if all requirements are completed."""
    if existing_requirements and len(steps_status) >= len(existing_requirements):
        # Check if any step is still pending
        if any(status == "pending" for status in steps_status):
            return

        if decision.next_worker != "end":
            logger.info("All steps completed. Forcing next_worker to 'end'.")
            decision.next_worker = "end"


async def supervisor_node(state: State) -> dict:
    """The supervisor node that routes to the correct worker or ends."""
    node_name = "supervisor"
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": node_name}
    )

    llm_profile = (
        settings.agent.supervisor.llm_profile or settings.llm.default_profile  # pylint: disable=no-member
    )
    llm = llm_factory(llm_profile, agent_name=node_name)
    # Initial Plan: If no worker is assigned (start of task), generate the plan.
    if state.get("next_worker", "") == "":
        llm_structured = llm.with_structured_output(SupervisorAgentPlanOutput)
    # Execution Loop: If a worker is assigned (or returning), continue execution.
    else:
        llm_structured = llm.with_structured_output(SupervisorAgentOutput)

    # Sanitize messages to remove orphaned ToolMessages caused by truncation
    conversation_messages = state["messages"]
    while conversation_messages and isinstance(conversation_messages[0], ToolMessage):
        conversation_messages = conversation_messages[1:]

    conversation_messages = sanitize_tool_calls(conversation_messages)

    steps_status = state.get("steps_status", [])

    system_prompt = SUPERVISOR_PROMPT
    system_prompt += f"\n\nTask ID: {state.get('task_id', 'N/A')}"
    existing_requirements = get_detected_requirements(state)
    if existing_requirements:
        req_strings = []
        for i, req in enumerate(existing_requirements):
            req_id = req.get("id")
            instruction = req.get("instruction")
            agent = req.get("assigned_agent")
            status = steps_status[i] if i < len(steps_status) else 'pending'
            req_strings.append(
                f"{i+1}. ID:{req_id} {instruction} (Agent: {agent}), status: {status}"
            )
        req_list = "\n".join(req_strings)
        system_prompt += (
            f"\n\n### ESTABLISHED PLAN\n{req_list}\n"
            "**INSTRUCTION**: The plan is LOCKED. You MUST follow these steps exactly in order.\n"
            "1. Identify the next step from the list above that has NOT been completed yet.\n"
            "2. Assign that step to the specified Agent.\n"
            "3. Do NOT change the plan or add new steps.\n"
            "4. **State Updates:** You MUST update `current_step_status` based on "
            "the results of the previous step.\n"
        )
    else:
        last_user_request = next(
        (
            m.content
            for m in reversed(conversation_messages)
            if isinstance(m, HumanMessage)
            and getattr(m, "name", None) != "supervisor"
        ),
        None,
        )
        if last_user_request:
            system_prompt += f"\n\n### LATEST USER REQUEST\n{last_user_request}\n"
            system_prompt += (
                "**FOCUS**: Plan based on this LATEST request. "
                "Ignore completed tasks from the chat history unless explicitly asked to retry."
            )

    messages = [SystemMessage(content=system_prompt)] + conversation_messages

    start_time = time.perf_counter()
    try:
        decision = await llm_structured.ainvoke(messages)
    except Exception as e:
        logger.error("LLM error: %s", e)
        # Fallback to prevent UnboundLocalError
        fallback_args = {
            "next_worker": "end",
            "response_to_user": f"The supervisor encountered an error: {str(e)}",
            "instructions_for_worker": "",
            "current_step_id": 0,
            "current_step_status": "failed"
        }
        
        if state.get("next_worker", "") == "":
            decision = SupervisorAgentPlanOutput(
                detected_requirements=[],
                **fallback_args
            )
        else:
            decision = SupervisorAgentOutput(**fallback_args)

    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": node_name}
    )

    if hasattr(decision, "detected_requirements") and decision.detected_requirements:
        telemetry.update_histogram(
            "agent.supervisor.plan.size", len(decision.detected_requirements)
        )


    existing_requirements = get_detected_requirements(state)
    update_step_status(decision, steps_status)

    # Check if supervisor wants to end but state shows pending steps
    enforce_plan(decision, existing_requirements, steps_status)
    # Check if task(s) are completed but the supervisor doesn't want to end
    check_completion(decision, existing_requirements, steps_status)

    nodes = state["response_metadata"].get("nodes", []) + [node_name]
    ai_message = AIMessage(
        content=decision.response_to_user
    )
    if hasattr(decision, "detected_requirements"):
        reqs_data = [req.model_dump(mode="json") for req in decision.detected_requirements]
        logger.info("requirements:\n%s", json.dumps(reqs_data, indent=2, default=str))
    logger.info("current id: %d", decision.current_step_id)
    logger.info("current step status: %s", decision.current_step_status)
    logger.info("steps status: %s", steps_status)
    logger.info("future skipped steps: %s", decision.skipped_step_ids)
    logger.info("next worker: %s", decision.next_worker)


    # Determine if we should update detected_requirements in the state.
    should_update_requirements = False
    if state.get("next_worker", "") == "":
        should_update_requirements = True

    output = {
        "response_metadata": {"nodes": nodes},
        "detected_requirements": state.get("detected_requirements", []),
        "steps_status": steps_status
    }
    if should_update_requirements:
        output["detected_requirements"] = decision.detected_requirements

    if decision.next_worker == "end":
        output["next_worker"] = "end"
        output["messages"] = [ai_message]
        return output

    context_message = HumanMessage(
        content=(
            "COMMAND FROM SUPERVISOR: "
            f"{decision.instructions_for_worker}\n"
            "**CONSTRAINT**: Focus ONLY on this instruction. Do NOT attempt to solve other parts of the user's request found in the chat history.\n"
            f"When finished or if you cannot proceed, use the '{SYSTEM_SUBMIT_WORK}' tool "
            "to report the result."
        ),
        name="supervisor"
    )

    output["next_worker"] = (
        decision.next_worker.value
        if hasattr(decision.next_worker, "value")
        else str(decision.next_worker)
    )
    output["messages"] = [context_message, ai_message]
    return output

async def system_node(state: State, tools: Sequence[BaseTool]) -> dict:
    """The system node that manages the infrastructure of the bot itself."""
    node_name = "system"
    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": node_name}
    )

    llm_profile = (
        settings.agent.system.llm_profile
        or settings.llm.default_profile  # pylint: disable=no-member
    )
    llm = llm_factory(llm_profile, agent_name=node_name)
    llm_with_tools = llm.bind_tools(tools)

    user_id = state.get("user_id", "User")
    prompt = f"{PLATFORM_PROMPT}\n{EXTENSION_PROMPT.format(user_id=user_id)}"
    messages = construct_llm_context(state, prompt)

    start_time = time.perf_counter()
    decision = await llm_with_tools.ainvoke(messages)
    duration = time.perf_counter() - start_time
    telemetry.update_histogram(
        "agent.llm.duration_seconds", duration, attributes={"agent": node_name}
    )

    nodes = state["response_metadata"].get("nodes", []) + [node_name]
    return {"messages": [decision], "response_metadata": {"nodes": nodes}}

async def auditor_node(state):
    if not settings.agent.auditor.enabled:
        return {}

    agent_name = "auditor"
    messages = state["messages"]
    original_request = "Unknown Request"
    last_request_index = -1

    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": agent_name}
    )

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
        if isinstance(m, ToolMessage)
        and i > last_request_index
        and m.name != SYSTEM_SUBMIT_WORK
    ]
    supervisor_messages = [
        str(m.content) for i, m in enumerate(messages)
        if isinstance(m, AIMessage)
        and i > last_request_index
        and m.content
    ]
    supervisor_log = "\n---\n".join(supervisor_messages)
    relevant_memories = state.get("relevant_memories", "")

    llm = llm_factory(llm_profile, agent_name=agent_name)
    llm_structured = llm.with_structured_output(AuditReport)
    audit = await llm_structured.ainvoke([
        SystemMessage(content=AUDITOR_PROMPT),
        HumanMessage(content=f"User Request: {original_request}"),
        HumanMessage(content=f"Memory Evidence (Past Knowledge): {relevant_memories}"),
        HumanMessage(content=f"Tool Evidence: {tool_outputs}"),
        HumanMessage(content=f"Supervisor Log: {supervisor_log}")
    ])

    nodes = state["response_metadata"].get("nodes", []) + [agent_name]

    telemetry.update_histogram("agent.auditor.score", audit.score)

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
            if isinstance(m, HumanMessage) and QA_REJECTION_PREFIX in str(m.content)
        ]
        if len(previous_rejections) >= 2:
            logger.info(
                "QA rejected %s times (Last score %s) Ending task.",
                len(previous_rejections),
                audit.score,
            )
            return {
                "messages": [
                    AIMessage(
                        content=f"QA Failed after {len(previous_rejections)} "
                        f"rejections. Finishing job (score: {audit.score})"
                    )
                ],
                "response_metadata": {"nodes": nodes}
            }

        first_failure = failed_verifications[0]
        feedback = (
            f"{QA_REJECTION_PREFIX}: {first_failure.missing_information}. "
            f"{first_failure.correction_instruction}"
        )
        logger.info(
            "QA Rejected with feedback (Score %s): %s",
            audit.score,
            feedback,
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

def human_node(state: State):
    """
    This node doesn't 'do' anything itself. 
    It acts as a placeholder for the Resume operation.
    When we resume, we will inject a HumanMessage pretending to be this node.
    """
    agent_name = "human"
    messages = state["messages"]

    telemetry.update_counter(
        "agent.invocations.total", attributes={"agent": agent_name}
    )

    return {"messages": [messages[-1]]}


def delete_tool_messages(messages: list) -> list:
    """Deletes tool messages from the conversation history."""
    return [RemoveMessage(id=m.id) for m in messages if isinstance(m, ToolMessage)]

async def manage_memory_node(state: State) -> dict:
    """Manages short-term and long-term memory for the conversation."""
    logger.info("Mem0: Managing memory...")
    mem0 = await get_mem0_client()
    messages = state["messages"]
    user_id = state["user_id"]

    # search
    try:
        start_time = time.perf_counter()
        memories = await mem0.search(
            messages[-1].content, user_id=user_id, limit=settings.mem0.limit
        )
        duration = time.perf_counter() - start_time
        telemetry.update_histogram(
            "agent.memory.operation.duration_seconds",
            duration,
            attributes={"operation": "search"},
        )
        memory_list = memories["results"]
        telemetry.update_counter("agent.memory.items.retrieved", len(memory_list))
        context = "Relevant information from previous conversations:\n"
        for memory in memory_list:
            context += f"- {memory['memory']}\n"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("mem0 err: %s", e)
        context = ""

    # Add to memory if we have a request-response pair
    human_messages = [
        msg for msg in state["messages"]
        if isinstance(msg, HumanMessage)
        and getattr(msg, "name", None) != "supervisor"
        and not str(msg.content).startswith(QA_REJECTION_PREFIX)
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
            if isinstance(msg, AIMessage)
            and not msg.tool_calls
            and not str(msg.content).startswith("QA Passed")
        ),
        None,
    )
    if last_human_message and last_ai_message:
        if contains_sensitive_data(last_human_message.content):
            logger.info("Mem0: Skipping memory add due to sensitive content.")
        else:
            interaction_messages = [
                {"role": "user", "content": last_human_message.content},
                {"role": "assistant", "content": last_ai_message.content},
            ]
            start_time = time.perf_counter()
            await mem0.add(
                messages=interaction_messages,
                user_id=user_id,
            )
            duration = time.perf_counter() - start_time
            telemetry.update_histogram(
                "agent.memory.operation.duration_seconds",
                duration,
                attributes={"operation": "add"},
            )
            logger.info("Mem0: Added interaction for user '%s'.", user_id)

    # Summarization if needed
    summarized_result, messages = await summarize_conversation(state)
    # delete tools message
    deleted_tool_messages = delete_tool_messages(state["messages"])
    if len(deleted_tool_messages) > 0:
        logger.info("Deleting %d tool messages.", len(deleted_tool_messages))
        messages.extend(deleted_tool_messages)

    return {"relevant_memories": context, "summary": summarized_result, "messages": messages}


async def astream_graph_updates(
    graph: StateGraph, user_input: str, user_id: str
) -> AsyncGenerator[AIMessage, Any]:
    """Streams updates from the graph execution."""
    config = {
        "configurable": {"thread_id": user_id},
        "recursion_limit": settings.graph.recursion_limit,
        "max_concurrency": settings.graph.max_concurrency,
    }

    current_state = await graph.aget_state(config)

    initial_input = {
        "messages": [{"role": "user", "content": user_input}],
        "user_id": user_id,
        "relevant_memories": "",  # Start with no memories, they will be fetched
        "next_worker": "",
        "response_metadata": {},
    }

    if not current_state.next:
        initial_input["task_id"] = str(uuid.uuid4())
        initial_input["detected_requirements"] = []
        initial_input["steps_status"] = []

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

    if await is_tool_authorized(tool_name, user_id):
        start_time = time.perf_counter()
        response = await handler(request)
        duration = time.perf_counter() - start_time
        telemetry.update_histogram(
            "agent.tool.duration_seconds", duration, attributes={"tool": tool_name}
        )
        return response

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
    """Routes based on auditor feedback."""
    last_msg = state["messages"][-1]
    if QA_REJECTION_PREFIX in last_msg.content:
        return "supervisor"
    else:
        return "approved"


def route_back_from_tool(state: State) -> str:
    """Routes back to the designated worker after a tool call."""
    # Check if the escalation tool was called
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            if message.name == SYSTEM_SUBMIT_WORK:
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
    """Checks if the guardrails has marked the conversation to end."""
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

    summary = state.get("summary", "")
    summary_message = []
    if summary:
        summary_message = [
            SystemMessage(content=f"**Previous Conversation Summary:**\n{summary}")
        ]

    # Sanitize messages to remove orphaned ToolMessages caused by truncation
    conversation_messages = state["messages"]
    while conversation_messages and isinstance(conversation_messages[0], ToolMessage):
        conversation_messages = conversation_messages[1:]

    conversation_messages = sanitize_tool_calls(conversation_messages)

    messages = (
        [SystemMessage(content=system_prompt)]
        + summary_message
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

def contains_sensitive_data(text: Any) -> bool:
    """
    Detects if the text likely contains sensitive data like tokens or keys.
    Returns True if a sensitive pattern is found, False otherwise.
    """
    # This pattern looks for keywords followed by a value, which is a strong
    # indicator of a secret.
    secret_pattern = r'(?i)\b(token|key|password|secret|authorization|bearer)\s*[:=]?\s+([^\s]+)'
    return bool(re.search(secret_pattern, str(text)))

def sanitize_tool_calls(messages: list) -> list:
    """
    Sanitizes messages to ensure AIMessages with tool_calls are followed by matching ToolMessages.
    It handles invalid_tool_calls and removes orphaned ToolMessages to prevent API errors (400).
    """
    sanitized = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]

        if isinstance(msg, AIMessage):
            if getattr(msg, "invalid_tool_calls", None):
                content = msg.content if msg.content else "..."
                sanitized.append(AIMessage(content=content, id=msg.id))
                i += 1
                continue

            if msg.tool_calls:
                expected_ids = {tc['id'] for tc in msg.tool_calls}

                tool_messages = []
                j = i + 1
                while j < n and isinstance(messages[j], ToolMessage):
                    tool_messages.append(messages[j])
                    j += 1

                found_ids = {tm.tool_call_id for tm in tool_messages}

                if expected_ids.issubset(found_ids):
                    sanitized.append(msg)
                    sanitized.extend(
                        [tm for tm in tool_messages if tm.tool_call_id in expected_ids]
                    )
                else:
                    content = msg.content if msg.content else "..."
                    sanitized.append(AIMessage(content=content, id=msg.id))

                i = j
                continue

        if isinstance(msg, ToolMessage):
            i += 1
        else:
            sanitized.append(msg)
            i += 1
    return sanitized

def detect_tool_loop(state, limit=3):
    """Detects if the agent is stuck in a loop calling the same tools."""
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
        logger.warning("Loop detected on tool: %s", last_call['name'])
        return True

    # Check for alternating loops (e.g., A, B, A, B)
    if len(tool_calls) >= 4:
        names = [t['name'] for t in tool_calls[-4:]]
        if names[0] == names[2] and names[1] == names[3]:
            return True

    return False

async def summarize_conversation(state: State):
    messages = state["messages"]
    current_summary = state.get("summary", "")

    # Use the supervisor's profile (or default) to count tokens via LangChain's native method.
    # The Supervisor is the bottleneck as it sees the full history for routing.
    # This handles ToolCall serialization correctly, avoiding "must contain a function key" errors.
    token_count_profile = (
        settings.agent.supervisor.llm_profile or settings.llm.default_profile
    )
    try:
        llm = llm_factory(token_count_profile, agent_name="memory")
        tokens = llm.get_num_tokens_from_messages(messages)
    except Exception as e:
        logger.warning("Failed to count tokens with profile '%s': %s", token_count_profile, e)
        return current_summary, []

    token_threshold = settings.memory.summarization.token_threshold
    messages_to_keep = settings.memory.summarization.keep_message
    if tokens < token_threshold and len(messages) < messages_to_keep:
        return current_summary, []

    if messages_to_keep > 0:
        older_messages = messages[:-messages_to_keep]
    else:
        older_messages = messages[:]

    if not older_messages:
        return current_summary, []

    # Generate Summary
    # Convert older messages to a string format for the prompt
    history_text = ""
    if current_summary:
        history_text += f"**Previous Summary:**\n{current_summary}\n\n"

    for m in older_messages:
        # skip tool messages as already we have supervisor messages
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, HumanMessage):
            if str(m.content).startswith(QA_REJECTION_PREFIX):
                continue
            history_text += f"User: {m.content}\n"
        elif isinstance(m, AIMessage) and m.content:
            history_text += f"Assistant: {m.content}\n"

    summary_text = current_summary
    summary_profile = settings.memory.summarization.llm_profile
    max_summary_token = int(token_threshold * 0.70)

    if summary_profile in settings.llm.profiles:
        try:
            llm = llm_factory(summary_profile, agent_name="memory")
            restricted_llm = llm.bind(max_tokens=max_summary_token)
            summary_response = await restricted_llm.ainvoke(
                SUMMARIZATION_PROMPT.format(
                    conversation_history=history_text,
                    max_summary_tokens=max_summary_token
                )
            )
            summary_text = summary_response.content
        except Exception as e:
            logger.error("Summarization failed: %s", e)

    delete_messages = [RemoveMessage(id=m.id) for m in older_messages]
    logger.info(
        "Thresholds triggered: token %d (threshold: %d), messages %d (keep: %d). "
        "Summarized %d messages.",
        tokens,
        token_threshold,
        len(messages),
        messages_to_keep,
        len(delete_messages),
    )

    return summary_text, delete_messages