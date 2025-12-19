import asyncio
import logging

import readline # noqa: F401, enables input history

from libs.logger import initialize_logger
initialize_logger()

from core.checkpoint import get_checkpointer # noqa: E402
from core.graph import run_graph, astream_graph_updates # noqa: E402
from services.telemetry.telemetry import Telemetry # noqa: E402
from core.memory import mem0_manager # noqa: E402

async def main():
    redis_client = None
    telemetry = None
    try:
        telemetry = Telemetry()
        async with get_checkpointer() as (checkpointer, redis_client):
            graph = await run_graph(checkpointer)            
            user_id = input("Please enter your username: ")
            if not user_id:
                print("Username cannot be empty. Exiting.")
                return
    
            while True:
                try:
                    user_input = input("User: ")
                    if not user_input.strip():
                        continue
                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        break
                    async for chunk in astream_graph_updates(graph, user_input, user_id):
                        chunk.pretty_print()
                        
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    
    finally:
        try:
            if telemetry:
                telemetry.shutdown()
            if redis_client:
                await redis_client.aclose()
                logging.info("Redis client closed.")
            await mem0_manager.close()
        except (asyncio.CancelledError, Exception):
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    
