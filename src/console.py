"""
This module provides a command-line interface for interacting with the RadOps assistant.
"""
import asyncio
import logging
from libs.logger import initialize_logger

initialize_logger()

from core.checkpoint import get_checkpointer
from core.graph import astream_graph_updates, run_graph
from core.memory import mem0_manager
from services.telemetry.telemetry import Telemetry



async def main():
    """
    The main function for the console application.
    """
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
                    async for chunk, metadata in astream_graph_updates(
                        graph, user_input, user_id
                    ):
                        print(
                            f"Assistants: {' -> '.join(metadata.get('nodes', []))}"
                        )
                        chunk.pretty_print()

                except (KeyboardInterrupt, EOFError):
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    # It's hard to predict what exceptions can be raised by the graph.
                    # For now, we catch all exceptions and log them.
                    logging.error("An error occurred: %s", e)
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
        except (asyncio.CancelledError, Exception) as e:
            # It's hard to predict what exceptions can be raised during cleanup.
            # For now, we catch all exceptions and log them.
            logging.error("An error occurred during cleanup: %s", e)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
