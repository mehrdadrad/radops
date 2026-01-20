# Playground CLI Guide

The Playground CLI (`src/playground/cli.py`) is a standalone WebSocket client designed to interact with the RadOps server. It mimics the behavior of a frontend client, allowing developers to test agent interactions, tool executions, and streaming responses directly from the terminal.

## Overview

- **File**: `src/playground/cli.py`
- **Protocol**: WebSocket
- **Dependencies**: `websocket-client` (install via `pip install websocket-client`)

## Usage

To start a chat session, run the script from the project root:

```bash
python src/playground/cli.py <user_id> [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `user_id` | `str` | Required | The unique identifier for the user (e.g., `user1`, `admin`). Used for role-based access control. |
| `--host` | `str` | `localhost` | The hostname or IP address where the RadOps server is running. |
| `--port` | `int` | `8005` | The port number of the RadOps server. |

### Examples

**1. Basic Connection**
Connect to the local server as `user123`:
```bash
python src/playground/cli.py user123
```

**2. Remote Server**
Connect to a server running on a specific host and port:
```bash
python src/playground/cli.py admin_user --host 10.0.0.5 --port 8080
```

## Interaction

- **Sending Messages**: Type your message and press Enter.
- **Streaming**: Responses from the agent are streamed in real-time.
- **History**: Use the Up/Down arrow keys to cycle through your message history.
- **Exiting**: Type `quit`, `exit`, `q`, or press `Ctrl+C` to end the session.