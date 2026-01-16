"""
This module provides a command-line interface for interacting with the RadOps backend
via WebSocket.
"""
import argparse
import readline
import sys
import threading

import websocket


def on_message(_ws, message):
    """
    Callback function to handle messages received from the server. It now
    handles redrawing the input prompt to avoid display issues.
    """
    sys.stdout.write("\r\033[K")  # Clear the current line
    if message == '\x03':
        print("\n--- task completed")
        sys.stdout.write("User: ")
        sys.stdout.flush()
    else:
        print(message)


def on_error(_ws, error):
    """
    Callback function to handle errors.
    """
    print(f"--- WebSocket Error: {error} ---", file=sys.stderr)


def on_open(ws):
    """
    Callback function when the connection is opened.
    This function will read from stdin and send messages to the server.
    """

    def send_input():
        try:
            while True:
                user_input = input("User: ")
                if not user_input.strip():
                    continue
                if user_input in ["q", "quit", "exit"]:
                    print("Goodbye!")
                    ws.close()
                    break
                ws.send(user_input)
        except (KeyboardInterrupt, EOFError):
            print()  # Print a newline for cleaner exit
            ws.close()

    threading.Thread(target=send_input, daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket chat client.")
    parser.add_argument(
        "user_id", type=str, help="The user ID to use for the chat session."
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="The server host."
    )
    parser.add_argument("--port", type=int, default=8005, help="The server port.")
    args = parser.parse_args()

    WS_URL = f"ws://{args.host}:{args.port}/ws/{args.user_id}"
    print(f"Connecting to {WS_URL}...")

    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )

    ws_app.run_forever()
