"""Slack integration module - Improved version."""
import logging
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import websockets
import websockets.exceptions
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from config.integrations import integration_settings as settings

# Constants
MAX_RETRY_ATTEMPTS = 2
MAX_CONNECTIONS = 100
MESSAGE_TIMEOUT = 300  # 5 minutes
HEALTH_CHECK_INTERVAL = 60  # 1 minute

logging.basicConfig(
    level=settings.slack.log_level.upper(),
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class ConnectionPoolFullError(Exception):
    """Raised when connection pool is at capacity."""


def validate_environment():
    """Validate that all required authentication settings are configured."""
    if settings.slack.auth_disabled:
        logger.warning("⚠️  auth_disabled is set to true - running without authentication!")
        logger.warning("⚠️  This should only be used in development environments!")
        return

    auth_method = settings.slack.auth_method.lower()

    if auth_method == "jwt":
        if not settings.slack.service_token:
            raise ValueError("Missing required configuration: slack.service_token")
        logger.info("Authentication method: JWT Bearer Token")
    elif auth_method == "api_key":
        if not settings.slack.service_api_key:
            raise ValueError("Missing required configuration: slack.service_api_key")
        logger.info("Authentication method: API Key")
    else:
        raise ValueError(
            f"Invalid auth_method: {auth_method}. Valid options: 'api_key' or 'jwt'"
        )


app = AsyncApp(token=settings.slack.bot_token)


class AsyncConnectionManager:
    """Manages WebSocket connections with connection pooling and health checks."""

    def __init__(self, max_connections: int = MAX_CONNECTIONS):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.max_connections = max_connections
        self._health_check_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the connection manager and health check loop."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """Stop the connection manager and close all connections."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        async with self.lock:
            for user_identifier in list(self.connections.keys()):
                await self._close_connection_internal(user_identifier)

    async def get_connection(self, user_identifier: str) -> Optional[Any]:
        """
        Get or create a WebSocket connection for a user.
        
        Args:
            user_identifier: User email or ID
            
        Returns:
            WebSocket connection or None if connection fails
            
        Raises:
            ConnectionPoolFullError: If max connections reached
        """
        async with self.lock:
            if user_identifier in self.connections:
                await self._refresh_connection_timer(user_identifier)
                return self.connections[user_identifier]['ws']

            if len(self.connections) >= self.max_connections:
                raise ConnectionPoolFullError(
                    f"Maximum connections ({self.max_connections}) reached"
                )

            return await self._create_connection(user_identifier)

    async def _refresh_connection_timer(self, user_identifier: str):
        """Refresh the inactivity timer for an existing connection."""
        old_timer = self.connections[user_identifier]['timer']
        if not old_timer.done():
            old_timer.cancel()
            try:
                await old_timer
            except asyncio.CancelledError:
                pass

        self.connections[user_identifier]['timer'] = asyncio.create_task(
            self._start_timer(user_identifier)
        )
        logger.info("Refreshed connection timer for user: %s", user_identifier)

    async def _create_connection(self, user_identifier: str) -> Optional[Any]:
        """Create a new WebSocket connection."""
        ws = None
        try:
            logger.info("Creating WebSocket connection for user: %s", user_identifier)

            connect_params = {"uri": f"{settings.slack.server_url}/ws/{user_identifier}"}

            if not settings.slack.auth_disabled:
                connect_params["extra_headers"] = self._get_auth_headers()

            ws = await websockets.connect(**connect_params)

            # Verify connection is usable
            await ws.ping()

            timer = asyncio.create_task(self._start_timer(user_identifier))
            self.connections[user_identifier] = {
                'ws': ws,
                'timer': timer,
                'created_at': time.time()
            }

            if settings.slack.auth_disabled:
                auth_status = "AUTH DISABLED"
            else:
                auth_status = settings.slack.auth_method
            logger.info(
                "WebSocket connection established for %s (%s)", user_identifier, auth_status
            )
            return ws

        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 401:
                logger.error("Authentication failed for user: %s", user_identifier)
                raise AuthenticationError("Invalid credentials") from e
            raise
        except Exception as e:
            logger.error("Connection error for %s: %s", user_identifier, e)
            if ws:
                await ws.close()
            return None

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on configured method."""
        auth_method = settings.slack.auth_method.lower()

        if auth_method == "jwt":
            if not settings.slack.service_token:
                raise AuthenticationError("service_token not configured for JWT")
            return {"Authorization": f"Bearer {settings.slack.service_token}"}
        elif auth_method == "api_key":
            if not settings.slack.service_api_key:
                raise AuthenticationError("service_api_key not configured")
            return {"X-API-Key": settings.slack.service_api_key}
        else:
            raise ValueError(f"Invalid auth_method: {auth_method}")

    async def _start_timer(self, user_identifier: str):
        """Start inactivity timer for a connection."""
        try:
            await asyncio.sleep(settings.slack.inactivity_timeout)
            await self.close_connection(user_identifier)
        except asyncio.CancelledError:
            pass

    async def close_connection(self, user_identifier: str):
        """Close the WebSocket connection for a specific user."""
        async with self.lock:
            await self._close_connection_internal(user_identifier)

    async def _close_connection_internal(self, user_identifier: str):
        """Internal method to close connection (must be called with lock held)."""
        if user_identifier not in self.connections:
            return

        connection_info = self.connections.pop(user_identifier)
        try:
            await connection_info['ws'].close()
            logger.info("Connection closed for user: %s", user_identifier)
        except Exception as e:
            logger.error("Error closing connection for %s: %s", user_identifier, e)
        finally:
            timer = connection_info['timer']
            if not timer.done():
                timer.cancel()
                try:
                    await timer
                except asyncio.CancelledError:
                    pass

    async def _health_check_loop(self):
        """Periodically check connection health."""
        while True:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                async with self.lock:
                    for user_identifier in list(self.connections.keys()):
                        try:
                            ws = self.connections[user_identifier]['ws']
                            await asyncio.wait_for(ws.ping(), timeout=5)
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.warning(
                                "Health check failed for %s: %s", user_identifier, e
                            )
                            await self._close_connection_internal(user_identifier)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop: %s", e)


manager = AsyncConnectionManager()


@app.event("message")
async def handle_message(body: Dict[str, Any], say: Callable, client: Any):
    """Handle incoming messages."""
    await handler(body, say, client)


@app.event("app_mention")
async def handle_mention(body: Dict[str, Any], say: Callable, client: Any):
    """Handle app mentions."""
    await handler(body, say, client)


async def handler(body: Dict[str, Any], say: Callable, client: Any):
    """Common handler for messages and mentions."""
    event = body["event"]
    user_id = event.get("user")
    text = event.get("text")
    channel_id = event.get("channel")
    message_ts = event.get("ts")

    # Ignore bot messages
    if event.get("bot_id"):
        return

    # Get user email
    user_email = await get_user_email(client, user_id)
    if not user_email:
        await say(
            "Could not retrieve email address. Please ensure the bot has `users:read.email` scope.",
            thread_ts=message_ts
        )
        return

    # Add reaction to acknowledge receipt
    await add_reaction(client, channel_id, message_ts, "eyes")

    # Process message with retry logic
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            await process_message(text, user_email, say, message_ts)
            break
        except websockets.exceptions.ConnectionClosed as e:
            await handle_connection_closed(e, user_email, say, message_ts, attempt)
            if should_stop_retry(e, attempt):
                break
        except ConnectionPoolFullError:
            await say(
                "Service is at capacity. Please try again in a moment.",
                thread_ts=message_ts
            )
            break
        except AuthenticationError:
            await say(
                "Authentication failed. Please contact your administrator.",
                thread_ts=message_ts
            )
            break
        except Exception as e:
            logger.exception("Unexpected error during message handling")
            await say(f"An error occurred: {type(e).__name__}", thread_ts=message_ts)
            await manager.close_connection(user_email)
            break


async def get_user_email(client: Any, user_id: str) -> Optional[str]:
    """Retrieve user email from Slack API."""
    try:
        user_info = await client.users_info(user=user_id)
        if user_info["ok"]:
            user_name = user_info["user"]["real_name"]
            user_email = user_info["user"]["profile"].get("email")
            logger.debug("Handling message from %s (%s)", user_name, user_id)
            return user_email
        else:
            logger.warning("Could not retrieve user info for %s", user_id)
            return None
    except Exception as e:
        logger.error("Error fetching user info: %s", e)
        return None


async def add_reaction(client: Any, channel_id: str, message_ts: str, emoji: str):
    """Add reaction to message with error handling."""
    try:
        await client.reactions_add(
            channel=channel_id,
            timestamp=message_ts,
            name=emoji
        )
    except Exception as e:
        logger.error("Error adding reaction: %s", e)


async def process_message(text: str, user_email: str, say: Callable, message_ts: str):
    """Process message through WebSocket connection."""
    ws = await manager.get_connection(user_email)
    if not ws:
        await say("Service unavailable.", thread_ts=message_ts)
        return

    await ws.send(text)

    try:
        async with asyncio.timeout(MESSAGE_TIMEOUT):
            while True:
                result = await ws.recv()
                if result == '\x03':  # End of transmission
                    break
                await say(result, thread_ts=message_ts)
    except asyncio.TimeoutError:
        logger.warning("Message processing timeout for user: %s", user_email)
        await say("Request timed out. Please try again.", thread_ts=message_ts)
        await manager.close_connection(user_email)


async def handle_connection_closed(
    error: websockets.exceptions.ConnectionClosed,
    user_email: str,
    say: Callable,
    message_ts: str,
    attempt: int
):
    """Handle WebSocket connection closed error."""
    logger.warning(
        "WebSocket connection closed: code=%s, reason=%s",
        error.code,
        error.reason
    )
    await manager.close_connection(user_email)

    if error.code == 1008:  # Policy Violation (authentication failure)
        auth_method = settings.slack.auth_method.lower()
        credential = 'service_token' if auth_method == 'jwt' else 'service_api_key'
        logger.error("Authentication failed - check slack.%s configuration", credential)
        await say(
            "Authentication failed. Please contact your administrator.",
            thread_ts=message_ts
        )
    elif error.code == 1012 and attempt == 0:  # Service restart - retry
        logger.info("Service restarting, retrying...")
    elif error.code == 1012:
        await say(
            "Service is restarting. Please try again in a moment.",
            thread_ts=message_ts
        )
    else:
        await say(f"Connection lost (code {error.code})", thread_ts=message_ts)


def should_stop_retry(error: websockets.exceptions.ConnectionClosed, attempt: int) -> bool:
    """Determine if retry should be stopped."""
    if error.code == 1008:  # Authentication failure
        return True
    if error.code == 1012 and attempt >= MAX_RETRY_ATTEMPTS - 1:  # Service restart, no more retries
        return True
    if error.code not in (1012,):  # Other errors, stop retrying
        return True
    return False


async def main():
    """Main entry point."""
    validate_environment()

    await manager.start()

    logger.info("Starting SLACK bolt with async handler...")
    handler = AsyncSocketModeHandler(app, settings.slack.app_token)

    try:
        await handler.start_async()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupt received. Shutting down...")
    finally:
        await handler.close_async()
        await manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass