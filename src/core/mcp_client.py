"""
MCP Client implementation with retry and backoff logic.
"""
import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import Field, create_model

logger = logging.getLogger(__name__)

# Suppress benign 404 warnings from MCP SSE client during shutdown
logging.getLogger("mcp.client.sse").setLevel(logging.ERROR)

class MCPClient:
    """
    A client for connecting to an MCP server with retry and backoff logic.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
    ):
        self.name = name
        self.config = config
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 5)
        self.persistent_interval = self.config.get("persistent_interval", 60)
        self.execution_timeout = self.config.get("execution_timeout", 60.0)
        self.connect_timeout = self.config.get("connect_timeout", 10.0)
        self.health_check_interval = self.config.get("health_check_interval", 10.0)

        self.session: Optional[ClientSession] = None
        self.tools: List[BaseTool] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tools_future = asyncio.Future()

    async def start(self):
        """Starts the connection loop in the background."""
        self._running = True
        self._task = asyncio.create_task(self._connection_manager())

    async def stop(self):
        """Stops the connection loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_tools(self) -> List[BaseTool]:
        """Returns the list of available tools, waiting if necessary."""
        if not self.tools and self._running:
            # Wait for initial connection if running but no tools yet
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._tools_future), timeout=5.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "[%s] get_tools timed out waiting for connection.", self.name
                )
                pass
        return self.tools

    async def _connection_manager(self):
        """Manages connection retries and persistence."""
        while self._running:
            try:
                await self._connect_with_retry()
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_msg = str(e)
                if hasattr(e, "exceptions"):
                    error_msg = f"{e} -> {[str(ex) for ex in e.exceptions]}"  # pylint: disable=no-member
                logger.error(
                    "[%s] Connection manager error: %s", self.name, error_msg
                )

            if not self._running:
                break

            logger.info(
                "[%s] Reconnecting in %ss...", self.name, self.persistent_interval
            )
            await asyncio.sleep(self.persistent_interval)

    async def _connect_with_retry(self):
        """Attempts to connect with immediate retries."""
        attempt = 0
        while attempt < self.retry_attempts and self._running:
            try:
                await self._connect_session()
                return
            except Exception as e:  # pylint: disable=broad-exception-caught
                attempt += 1
                error_msg = str(e)
                if hasattr(e, "exceptions"):
                    error_msg = f"{e} -> {[str(ex) for ex in e.exceptions]}"  # pylint: disable=no-member

                logger.warning(
                    "[%s] Connection attempt %s/%s failed: %s",
                    self.name,
                    attempt,
                    self.retry_attempts,
                    error_msg,
                )
                if attempt < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Let the outer loop handle the persistent wait
                    raise e

    async def _connect_session(self):
        """Establishes the MCP session and waits until it closes."""
        transport = self.config.get("transport", "stdio")

        if transport == "stdio":
            env = os.environ.copy()
            if env_config := self.config.get("env"):
                env.update({k: str(v) for k, v in env_config.items()})

            server_params = StdioServerParameters(
                command=self.config.get("command"),
                args=self.config.get("args", []),
                env=env,
            )
            client_context = stdio_client(server_params)
        elif transport in ("streamable_http", "sse"):
            if not (url := self.config.get("url")):
                raise ValueError(f"URL is required for {transport} transport")

            kwargs = {
                "url": url,
                "headers": self.config.get("headers") or {},
                "timeout": self.connect_timeout,
            }
            client_factory = (
                streamablehttp_client if transport == "streamable_http" else sse_client
            )
            client_context = client_factory(**kwargs)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        logger.info("[%s] Connecting to MCP server...", self.name)
        try:
            async with client_context as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    self.session = session
                    await asyncio.wait_for(session.initialize(), timeout=self.connect_timeout)

                    # Load tools
                    result = await asyncio.wait_for(session.list_tools(), timeout=self.connect_timeout)
                    self.tools = self._process_tools(result.tools)

                    if not self._tools_future.done():
                        self._tools_future.set_result(True)

                    logger.info(
                        "[%s] Connected. Loaded %s tools.", self.name, len(self.tools)
                    )

                    # Keep session alive and monitor health
                    await self._monitor_health(session)

                    # If the loop exits but we are still running,
                    # the session closed cleanly (e.g. EOF)
                    if self._running:
                        raise ConnectionError("Connection closed unexpectedly")
        finally:
            logger.info("[%s] Session disconnected.", self.name)
            self.session = None
            self.tools = []
            # Reset future for next try
            if self._tools_future.done():
                self._tools_future = asyncio.Future()

    async def _monitor_health(self, session: ClientSession):
        """Monitors the health of the session."""
        while self._running:
            await asyncio.sleep(self.health_check_interval)
            try:
                await asyncio.wait_for(session.list_tools(), timeout=self.connect_timeout)
            except asyncio.CancelledError as exc:
                if self._running:
                    logger.warning(
                        "[%s] Health check cancelled (connection dropped).",
                        self.name,
                    )
                    raise ConnectionError(
                        "Connection dropped (Cancelled)"
                    ) from exc
                raise
            except Exception as e:
                logger.warning("[%s] Health check failed: %s", self.name, e)
                raise e

    def _process_tools(self, mcp_tools: List[Any]) -> List[BaseTool]:
        """Converts MCP tools to LangChain tools and applies prefix."""
        langchain_tools = []
        prefix = f"{self.name}__"

        for tool in mcp_tools:
            tool_name = f"{prefix}{tool.name}"
            current_tool_name = tool.name
            args_schema = self._create_args_schema(tool_name, tool.inputSchema)

            tool_coro = self._create_tool_func(tool_name, current_tool_name)
            langchain_tools.append(
                StructuredTool.from_function(
                    func=None,
                    coroutine=tool_coro,
                    name=tool_name,
                    description=tool.description or "",
                    args_schema=args_schema,
                )
            )

        return langchain_tools

    def _create_tool_func(self, tool_name: str, current_tool_name: str):
        """Creates a tool function with captured variables."""
        async def _tool_func(**kwargs):
            kwargs = sanitize_kwargs(kwargs)
            if not self.session:
                raise RuntimeError(f"Tool {tool_name} is not connected.")
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(current_tool_name, arguments=kwargs),
                    timeout=self.execution_timeout
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Tool {tool_name} execution timed out."
                ) from exc
            output = []
            if hasattr(result, "content"):
                for item in result.content:
                    if item.type == "text":
                        output.append(item.text)
            if result.isError:
                logger.warning("Tool %s failed: %s", tool_name, output)
            return "\n".join(output)
        return _tool_func

    def _create_args_schema(self, tool_name: str, schema: Dict[str, Any]) -> Any:
        """Creates a Pydantic model from a JSON schema."""
        fields = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for field_name, field_def in properties.items():
            field_type = type_map.get(field_def.get("type", "string"), str)
            description = field_def.get("description", "")

            if field_name in required:
                fields[field_name] = (field_type, Field(description=description))
            else:
                fields[field_name] = (
                    Optional[field_type],
                    Field(default=None, description=description),
                )

        return create_model(f"{tool_name.replace('-', '_')}Args", **fields)


def sanitize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes tool arguments by parsing stringified JSON.

    Some LLMs may incorrectly pass JSON objects or arrays as strings.
    This function iterates through the arguments and attempts to parse
    any string that looks like a JSON object or array.

    Args:
        kwargs: The original dictionary of arguments from the LLM.

    Returns:
        A new dictionary with stringified JSON values parsed into
        Python objects (dicts or lists).
    """
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, str) and (
            (v.strip().startswith("{") and v.strip().endswith("}"))
            or (v.strip().startswith("[") and v.strip().endswith("]"))
        ):
            try:
                result[k] = json.loads(v)
                logger.debug("Parsed JSON string for argument '%s'", k)
                continue
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON string for argument '%s'", k)
        result[k] = v
    return result
