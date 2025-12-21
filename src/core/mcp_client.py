"""
MCP Client implementation with retry and backoff logic.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import Field, create_model

logger = logging.getLogger(__name__)

class MCPClient:
    """
    A client for connecting to an MCP server with retry and backoff logic.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        retry_attempts: int = 3,
        retry_delay: int = 5,
        persistent_interval: int = 60,
    ):
        self.name = name
        self.config = config
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.persistent_interval = persistent_interval

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
                pass
        return self.tools

    async def _connection_manager(self):
        """Manages connection retries and persistence."""
        while self._running:
            try:
                await self._connect_with_retry()
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, "exceptions"):
                    error_msg = f"{e} -> {[str(ex) for ex in e.exceptions]}"
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
            except Exception as e:
                attempt += 1
                error_msg = str(e)
                if hasattr(e, "exceptions"):
                    error_msg = f"{e} -> {[str(ex) for ex in e.exceptions]}"

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
            command = self.config.get("command")
            args = self.config.get("args", [])
            env_config = self.config.get("env", {})

            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_config.items()})

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
            )
            client_context = stdio_client(server_params)
        elif transport == "streamable_http":
            url = self.config.get("url")
            headers = self.config.get("headers") or {}
            timeout = self.config.get("connect_timeout", 60.0)
            if not url:
                raise ValueError(f"URL is required for {transport} transport")
            client_context = streamablehttp_client(
                url=url, headers=headers, timeout=timeout
            )
        elif transport == "sse":
            url = self.config.get("url")
            headers = self.config.get("headers") or {}
            timeout = self.config.get("connect_timeout", 60.0)
            if not url:
                raise ValueError(f"URL is required for {transport} transport")
            client_context = sse_client(
                url=url, headers=headers, timeout=timeout
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        logger.info("[%s] Connecting to MCP server...", self.name)
        try:
            async with client_context as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    self.session = session
                    await asyncio.wait_for(session.initialize(), timeout=10.0)

                    # Load tools
                    result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                    self.tools = self._process_tools(result.tools)

                    if not self._tools_future.done():
                        self._tools_future.set_result(True)

                    logger.info(
                        "[%s] Connected. Loaded %s tools.", self.name, len(self.tools)
                    )

                    # Keep session alive and monitor health
                    while self._running:
                        await asyncio.sleep(10)
                        try:
                            await asyncio.wait_for(session.list_tools(), timeout=5.0)
                        except asyncio.CancelledError:
                            if self._running:
                                logger.warning(
                                    "[%s] Health check cancelled (connection dropped).",
                                    self.name,
                                )
                                raise ConnectionError("Connection dropped (Cancelled)")
                            raise
                        except Exception as e:
                            logger.warning("[%s] Health check failed: %s", self.name, e)
                            raise e

                    # If the loop exits but we are still running, the session closed cleanly (e.g. EOF)
                    if self._running:
                        raise ConnectionError("Connection closed unexpectedly")
        finally:
            logger.info("[%s] Session disconnected.", self.name)
            self.session = None
            self.tools = []
            # Reset future for next try
            if self._tools_future.done():
                self._tools_future = asyncio.Future()

    def _process_tools(self, mcp_tools: List[Any]) -> List[BaseTool]:
        """Converts MCP tools to LangChain tools and applies prefix."""
        langchain_tools = []
        prefix = f"{self.name}__"

        for tool in mcp_tools:
            tool_name = f"{prefix}{tool.name}"
            current_tool_name = tool.name
            args_schema = self._create_args_schema(tool_name, tool.inputSchema)

            langchain_tools.append(
                StructuredTool.from_function(
                    func=None,
                    coroutine=self._create_tool_func(tool_name, current_tool_name),
                    name=tool_name,
                    description=tool.description or "",
                    args_schema=args_schema,
                )
            )

        return langchain_tools

    def _create_tool_func(self, tool_name: str, current_tool_name: str):
        """Creates a tool function with captured variables."""
        async def _tool_func(**kwargs):
            if not self.session:
                raise RuntimeError(f"Tool {tool_name} is not connected.")
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(current_tool_name, arguments=kwargs),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Tool {tool_name} execution timed out.")
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