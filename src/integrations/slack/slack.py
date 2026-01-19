"""Slack integration module."""
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
    level=settings.slack.log_level.upper(),
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


# Validate required environment variables on startup
def validate_environment():
    """Validate that all required authentication settings are configured."""
    if settings.slack.auth_disabled:
        logger.warning("⚠️  auth_disabled is set to true - running without authentication!")
        logger.warning("⚠️  This should only be used in development environments!")
        return

    # Check for required auth credentials based on method
    auth_method = settings.slack.auth_method.lower()

    if auth_method == "jwt":
        if not settings.slack.service_token:
            logger.error("Missing required configuration: slack.service_token")
            sys.exit(1)
        logger.info("Authentication method: JWT Bearer Token")
    elif auth_method == "api_key":
        if not settings.slack.service_api_key:
            logger.error("Missing required configuration: slack.service_api_key")
            sys.exit(1)
        logger.info("Authentication method: API Key")
    else:
        logger.error(f"Invalid auth_method: {auth_method}")
        logger.error("Valid options: 'api_key' or 'jwt'")
        sys.exit(1)


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
                logger.info("Refreshed connection timer for user")
                return self.connections[user_id]['ws']

            try:
                logger.info("Creating WebSocket connection for user")

                # Prepare connection parameters
                connect_params = {"uri": f"{settings.slack.server_url}/ws/{user_id}"}

                if not settings.slack.auth_disabled:
                    # Add authentication header based on method
                    auth_method = settings.slack.auth_method.lower()

                    if auth_method == "jwt":
                        if settings.slack.service_token:
                            # JWT uses Authorization: Bearer <token>
                            connect_params["extra_headers"] = {
                                "Authorization": f"Bearer {settings.slack.service_token}"
                            }
                        else:
                            logger.error("service_token not configured")
                            return None
                    else:  # api_key
                        if settings.slack.service_api_key:
                            # API Key uses X-API-Key header
                            connect_params["extra_headers"] = {
                                "X-API-Key": settings.slack.service_api_key
                            }
                        else:
                            logger.error("service_api_key not configured")
                            return None

                try:
                    ws = await websockets.connect(**connect_params)
                    if settings.slack.auth_disabled:
                        logger.info("WebSocket connection established (AUTH DISABLED)")
                    else:
                        logger.info(
                            "WebSocket connection authenticated successfully (%s)", auth_method
                        )
                except TypeError as e:
                    if "unexpected keyword argument 'extra_headers'" in str(e):
                        logger.error(
                            "The 'websockets' library version is too old and "
                            "doesn't support authentication. "
                            "Please upgrade: pip install --upgrade websockets>=10.0"
                        )
                        raise RuntimeError(
                            "Authentication not supported by websockets library") from e
                    else:
                        raise

                self.connections[user_id] = {
                    'ws': ws,
                    'timer': asyncio.create_task(self._start_timer(user_id))
                }
                return ws
            except Exception as e:
                logger.error(f"Connection error: {e}")
                return None

    async def _start_timer(self, user_id):
        await asyncio.sleep(settings.slack.inactivity_timeout)
        await self.close_connection(user_id)

    async def close_connection(self, user_id):
        """Closes the WebSocket connection for a specific user."""
        async with self.lock:
            if user_id in self.connections:
                connection_info = self.connections.pop(user_id)
                try:
                    await connection_info['ws'].close()
                    logger.info("Connection closed for user due to inactivity")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error closing connection: %s", e)
                finally:
                    if not connection_info['timer'].done():
                        connection_info['timer'].cancel()
                    logger.info("Removed user from connection manager")


manager = AsyncConnectionManager()


@app.event("message")
async def handle_message(body, say, client):
    """Handles incoming messages."""
    await handler(body, say, client)


@app.event("app_mention")
async def handle_mention(body, say, client):
    """Handles app mentions."""
    await handler(body, say, client)


async def handler(body, say, client):
    """Common handler for messages and mentions."""
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
            logger.debug(f"Handling message from {user_name} ({user_id})")
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
            logger.warning(f"WebSocket connection closed: code={e.code}, reason={e.reason}")
            await manager.close_connection(user_email)

            if e.code == 1008:  # Policy Violation (authentication failure)
                auth_method = settings.slack.auth_method.lower()
                credential = 'service_token' if auth_method == 'jwt' else 'service_api_key'
                logger.error(f"Authentication failed - check slack.{credential} configuration")
                await say(
                    "Authentication failed. Please contact your administrator.",
                    thread_ts=event.get("ts")
                )
                break
            elif e.code == 1012 and attempt == 0:  # Service restart - retry
                logger.info("Service restarting, retrying...")
                continue
            elif e.code == 1012:
                await say(
                    "Service is restarting. Please try again in a moment.",
                    thread_ts=event.get("ts")
                )
                break
            else:
                await say(f"Connection lost (code {e.code})", thread_ts=event.get("ts"))
                break
        except Exception as e:
            logger.error(f"Error during message handling: {e}")
            await say(f"Error: {e}", thread_ts=event.get("ts"))
            await manager.close_connection(user_email)
            break


async def main():
    validate_environment()
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