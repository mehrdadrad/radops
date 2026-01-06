import logging
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import websockets
import websockets.exceptions
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from config.integrations import integration_settings as settings

logging.basicConfig(
    level=settings.slack.log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s', force=True
)
logger = logging.getLogger(__name__)


app = AsyncApp(
    token=settings.slack.bot_token,
)


class AsyncConnectionManager:
    def __init__(self):
        self.connections = {}
        self.lock = asyncio.Lock()

    async def get_connection(self, user_id):
        async with self.lock:
            if user_id in self.connections:
                # Cancel existing timer and start a new one
                self.connections[user_id]['timer'].cancel()
                self.connections[user_id]['timer'] = asyncio.create_task(self._start_timer(user_id))
                logger.info(f"Refreshed connection timer for user {user_id}")
                return self.connections[user_id]['ws']

            try:
                logger.info(f"Creating async connection for user {user_id}")
                ws = await websockets.connect(f"ws://localhost:8005/ws/{user_id}")
                self.connections[user_id] = {
                    'ws': ws,
                    'timer': asyncio.create_task(self._start_timer(user_id))
                }
                return ws
            except Exception as e:
                logger.error(f"Async connection error for user {user_id}: {e}")
                return None

    async def _start_timer(self, user_id):
        await asyncio.sleep(60.0)
        await self.close_connection(user_id)

    async def close_connection(self, user_id):
        async with self.lock:
            if user_id in self.connections:
                connection_info = self.connections.pop(user_id)
                try:
                    await connection_info['ws'].close()
                    logger.info(f"Async connection closed for user {user_id} due to inactivity.")
                except Exception as e:
                    logger.error(f"Error closing connection for {user_id}: {e}")
                finally:
                    if not connection_info['timer'].done():
                        connection_info['timer'].cancel()
                    logger.info(f"Removed user {user_id} from async manager.")


manager = AsyncConnectionManager()


@app.event("message")
async def handle_message(body, say, client):
    await handler(body, say, client)


@app.event("app_mention")
async def handle_mention(body, say, client):
    await handler(body, say, client)


async def handler(body, say, client):
    event = body["event"]
    user_id = event.get("user")
    text = event.get("text")
    channel_id = event.get("channel")
    message_ts = event.get("ts")

    if event.get("bot_id"):
        return

    user_email = None
    try:
        user_info = await client.users_info(user=user_id)
        if user_info["ok"]:
            user_name = user_info["user"]["real_name"]
            user_email = user_info["user"]["profile"].get("email")
            logger.info(f"Handling message from {user_name} ({user_id})")
        else:
            logger.warning(f"Could not retrieve user info for {user_id}")
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")

    try:
        await client.reactions_add(
            channel=channel_id,
            timestamp=message_ts,
            name="eyes"
        )
    except Exception as e:
        logger.error(f"Error adding reaction: {e}")

    if not user_email:
        await say(
            "Could not retrieve email address. Please ensure the bot has `users:read.email` scope.",
            thread_ts=event.get("ts")
        )
        return

    for attempt in range(2):
        ws = await manager.get_connection(user_email)
        if not ws:
            await say("Service unavailable.", thread_ts=event.get("ts"))
            return

        try:
            await ws.send(text)
            while True:
                result = await ws.recv()
                if result == '\x03':
                    break
                await say(result, thread_ts=event.get("ts"))
            break
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed for {user_email}: {e}")
            await manager.close_connection(user_email)
            if e.code == 1012 and attempt == 0:
                continue
            elif e.code == 1012:
                await say("Service is restarting. Please try again in a moment.", thread_ts=event.get("ts"))
            else:
                await say(f"Connection lost: {e}", thread_ts=event.get("ts"))
            break
        except Exception as e:
            await say(f"Error: {e}", thread_ts=event.get("ts"))
            await manager.close_connection(user_email)
            break


async def main():
    logger.info("Starting SLACK bolt with async handler...")
    handler = AsyncSocketModeHandler(
        app,
        settings.slack.app_token
    )
    try:
        await handler.start_async()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupt received. Shutting down...")
    finally:
        await handler.close_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass