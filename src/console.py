"""
This module provides a command-line interface for interacting with the RadOps assistant.
"""
import asyncio
import argparse
import logging
import warnings
from libs.logger import initialize_logger

initialize_logger()

from core.checkpoint import get_checkpointer
from core.graph import astream_graph_updates, run_graph
from core.llm import close_shared_client
from core.memory import mem0_manager
from services.telemetry.telemetry import telemetry
from tools import ToolRegistry

# Suppress Weaviate ResourceWarning on shutdown
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*The connection to Weaviate was not closed properly.*")

async def ainput(prompt: str = "") -> str:
    """Async wrapper for input function."""
    return await asyncio.to_thread(input, prompt)

async def main(skip_initial_sync: bool = False):
    """
    The main function for the console application.
    """
    redis_client = None
    tool_registry = None
    try:
        async with get_checkpointer() as (checkpointer, redis_client):
            tool_registry = ToolRegistry(checkpointer=checkpointer, skip_initial_sync=skip_initial_sync)
            graph = await run_graph(checkpointer, tool_registry=tool_registry)
            user_id = await ainput("Please enter your username: ")
            if not user_id:
                print("Username cannot be empty. Exiting.")
                return

            while True:
                try:
                    user_input = await ainput("User: ")
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
                    logging.error("An error occurred: %s", e)
                    continue
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        logging.error("Fatal error: %s", e)

    finally:
        try:
            telemetry.shutdown()
            if redis_client:
                await redis_client.aclose()
                logging.info("Redis client closed.")
            if tool_registry:
                await tool_registry.close()
                logging.info("Tool registry closed.")
            await mem0_manager.close()
            await close_shared_client()
        except (asyncio.CancelledError, Exception) as e:
            logging.error("An error occurred during cleanup: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RadOps Console")
    parser.add_argument("--skip-initial-sync", action="store_true", help="Skip the initial vector store synchronization.")
    args = parser.parse_args()

    try:
        asyncio.run(main(skip_initial_sync=args.skip_initial_sync))
    except KeyboardInterrupt:
        pass
