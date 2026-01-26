"""
This module implements a FastAPI server for the RadOps assistant.
"""
import logging
import sys
import warnings
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status

from libs.logger import initialize_logger

initialize_logger()

from config.server import server_settings
from core.auth import get_user_role
from core.checkpoint import get_checkpointer
from core.llm import close_shared_client
from core.graph import astream_graph_updates, run_graph
from core.memory import mem0_manager
from services.telemetry.telemetry import telemetry
from registry.tools import ToolRegistry
from libs.status_generator import StatusGenerator


# Suppress Weaviate ResourceWarning on shutdown
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=".*The connection to Weaviate was not closed properly.*",
)


app = FastAPI(
    title="RadOps Chatbot Server",
    version="1.0",
    description="A simple API server for interacting with RadOps.",
)


def authenticate_websocket(websocket: WebSocket) -> dict:
    """
    Authenticates WebSocket connection using headers.
    Returns service account info or raises WebSocketDisconnect.
    """
    if server_settings.auth_disabled:
        return {"sub": "dev-service", "role": "service", "type": "development"}

    # Extract headers from WebSocket scope
    headers = dict(websocket.scope.get("headers", []))

    # Check for API Key in X-API-Key header
    api_key = headers.get(b"x-api-key")
    if api_key:
        api_key = api_key.decode("utf-8")
        service_api_key = server_settings.service_api_key
        if service_api_key and api_key == service_api_key:
            return {"sub": "radops-service", "role": "service", "type": "api_key"}

    # Check for JWT in Authorization header
    auth_header = headers.get(b"authorization")
    if auth_header:
        auth_header = auth_header.decode("utf-8")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            service_token = server_settings.service_token
            if service_token and token == service_token:
                return {"sub": "radops-service", "role": "service", "type": "jwt"}

    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Manages the application's lifespan, initializing resources on startup
    and cleaning them up on shutdown.
    """
    logging.info("Application startup: Initializing resources.")
    checkpointer, redis_client, tool_registry = None, None, None
    try:
        async with get_checkpointer() as (cp, rc):
            try:
                checkpointer, redis_client = cp, rc
                tool_registry = ToolRegistry(
                    checkpointer=checkpointer, skip_initial_sync=server_settings.skip_initial_sync
                )
                fastapi_app.state.graph = await run_graph(checkpointer, tool_registry=tool_registry)
                fastapi_app.state.checkpointer = checkpointer
                fastapi_app.state.redis_client = redis_client
                yield
            except Exception as e:
                logging.error("Fatal error during startup: %s", e)
                sys.exit(1)
    finally:
        logging.info("Application shutdown: Cleaning up resources.")
        if redis_client:
            await redis_client.aclose()
            logging.info("Redis client closed.")
        if tool_registry:
            await tool_registry.close()
            logging.info("Tool registry closed.")
        await mem0_manager.close()
        logging.info("mem0_manager closed.")
        await close_shared_client()
        logging.info("Shared LLM client closed.")
        telemetry.shutdown()
        logging.info("Telemetry shutdown.")


app.router.lifespan_context = lifespan


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
):
    """
    Handles WebSocket connections for chat, taking user_id from the path.
    """
    # Authenticate BEFORE accepting the connection
    try:
        service_account = authenticate_websocket(websocket)
    except HTTPException as e:
        logging.warning(
            "WebSocket authentication failed for user_id=%s: %s",
            user_id,
            e.detail
        )
        await websocket.close(code=1008, reason="Authentication failed")
        return

    # Now accept the connection
    await websocket.accept()

    logging.info(
        "WebSocket authenticated: service=%s, auth_type=%s, user_id=%s",
        service_account.get("sub"),
        service_account.get("type"),
        user_id
    )

    if not await get_user_role(user_id):
        logging.warning("Connection rejected for unknown user: %s", user_id)
        await websocket.send_text("Error: Access denied.")
        await websocket.close(code=1008)
        return

    graph = websocket.app.state.graph
    logging.info("WebSocket connection established for user_id: %s", user_id)

    try:
        async for user_input in websocket.iter_text():
            logging.info("Received message from %s", user_id)
            async for chunk, metadata in astream_graph_updates(
                graph, user_input, user_id
            ):
                message = None
                if chunk.tool_calls:
                    tool_name = chunk.tool_calls[0].get("name", "unknown")
                    agent_name = "unknown"
                    if metadata and "nodes" in metadata and metadata["nodes"]:
                        agent_name = metadata["nodes"][-1]

                    if server_settings.plain_message:
                        message = f"Agent: {agent_name} -> Running tool: {tool_name}"
                    else:
                        status = StatusGenerator.parse_tool_call(tool_name, {})
                        message = f"{status}"
                else:
                    message = str(chunk.content)
                await websocket.send_text(message)
            await websocket.send_text("\x03")
    except WebSocketDisconnect:
        logging.info(
            "WebSocket connection closed cleanly for user_id: %s", user_id
        )
    except Exception as e:
        logging.error(
            "An error occurred for user_id %s: %s", user_id, e,
        )
        await websocket.close(code=1011, reason="Internal Server Error")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RadOps Chatbot Server")
    parser.add_argument("--host", default=server_settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=server_settings.port, help="Port to bind to")
    parser.add_argument(
        "--skip-initial-sync",
        action="store_true",
        help="Skip the initial vector store synchronization.",
    )
    parser.add_argument(
        "--plain-message",
        action="store_true",
        help="Use plain text messages for tool calls.",
    )
    args = parser.parse_args()

    if args.skip_initial_sync:
        server_settings.skip_initial_sync = True
    if args.plain_message:
        server_settings.plain_message = True

    uvicorn.run(
        app, host=args.host, port=args.port, log_config=None, ws="wsproto"
    )
    