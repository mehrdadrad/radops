import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from core.graph import run_graph, astream_graph_updates
from core.checkpoint import get_checkpointer
from services.telemetry.telemetry import Telemetry
from core.memory import mem0_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

app = FastAPI(
    title="Chatbot Server",
    version="1.0",
    description="A simple API server for interacting with a net assistant."
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, initializing resources on startup
    and cleaning them up on shutdown.
    """
    logging.info("Application startup: Initializing resources.")
    telemetry = Telemetry()
    checkpointer, redis_client = None, None
    try:
        async with get_checkpointer() as (cp, rc):
            checkpointer, redis_client = cp, rc
            app.state.graph = await run_graph(checkpointer)
            app.state.checkpointer = checkpointer
            app.state.redis_client = redis_client
            yield
    finally:
        logging.info("Application shutdown: Cleaning up resources.")
        if redis_client:
            await redis_client.aclose()
            logging.info("Redis client closed.")
        await mem0_manager.close()
        logging.info("mem0_manager closed.")
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
    logging.info(f"WebSocket connection established for user_id: {user_id}")

    try:
        async for user_input in websocket.iter_text():
            logging.info(f"Received message from {user_id}: {user_input}")
            async for chunk in astream_graph_updates(graph, user_input, user_id):
                message = None
                if chunk.tool_calls:
                    tool_name = chunk.tool_calls[0].get('name', 'unknown')
                    message = f"Running tool: {tool_name}"
                else:
                    message = str(chunk.content)     
                await websocket.send_text(message)
            await websocket.send_text('\x03')    
    except WebSocketDisconnect:
        logging.info(f"WebSocket connection closed cleanly for user_id: {user_id}")
    except Exception as e:
        logging.error(f"An error occurred for user_id {user_id}: {e}", exc_info=True)
        await websocket.close(code=1011, reason="Internal Server Error")
    finally:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, log_config=None, ws="wsproto")