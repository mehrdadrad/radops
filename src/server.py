"""
This module implements a FastAPI server for the RadOps assistant.
"""
import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from libs.logger import initialize_logger

initialize_logger()

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from core.checkpoint import get_checkpointer
from core.llm import close_shared_client
from core.graph import astream_graph_updates, run_graph
from core.memory import mem0_manager
from services.telemetry.telemetry import telemetry
from tools import ToolRegistry
from libs.status_generator import StatusGenerator

USE_PLAIN_MESSAGE = os.getenv("PLAIN_MESSAGE", "false").lower() == "true" or "--plain-message" in sys.argv
SKIP_INITIAL_SYNC = os.getenv("SKIP_INITIAL_SYNC", "false").lower() in ("true", "1", "t") or "--skip-initial-sync" in sys.argv

# Suppress Weaviate ResourceWarning on shutdown
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*The connection to Weaviate was not closed properly.*")


app = FastAPI(
    title="RadOps Chatbot Server",
    version="1.0",
    description="A simple API server for interacting with RadOps.",
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
                tool_registry = ToolRegistry(checkpointer=checkpointer, skip_initial_sync=SKIP_INITIAL_SYNC)
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
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    Handles WebSocket connections for chat, taking user_id from the path.
    """
    await websocket.accept()
    graph = websocket.app.state.graph
    logging.info("WebSocket connection established for user_id: %s", user_id)

    try:
        async for user_input in websocket.iter_text():
            logging.info("Received message from %s: %s", user_id, user_input)
            async for chunk, metadata in astream_graph_updates(
                graph, user_input, user_id
            ):
                message = None
                if chunk.tool_calls:
                    tool_name = chunk.tool_calls[0].get("name", "unknown")
                    agent_name = "unknown"
                    if metadata and "nodes" in metadata and metadata["nodes"]:
                        agent_name = metadata["nodes"][-1]

                    if USE_PLAIN_MESSAGE:
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
            "An error occurred for user_id %s: %s", user_id, e, exc_info=True
        )
        await websocket.close(code=1011, reason="Internal Server Error")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RadOps Chatbot Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind to")
    parser.add_argument("--skip-initial-sync", action="store_true", help="Skip the initial vector store synchronization.")
    parser.add_argument("--plain-message", action="store_true", help="Use plain text messages for tool calls.")
    args = parser.parse_args()

    uvicorn.run(
        app, host=args.host, port=args.port, log_config=None, ws="wsproto"
    )