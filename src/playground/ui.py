"""
This module provides a Streamlit UI for interacting with the RadOps backend
via WebSocket.

Usage:
    streamlit run src/playground/ui.py
"""
import streamlit as st
import websocket

st.set_page_config(page_title="RadOps Playground", page_icon="💬")
st.title("RadOps Playground")

# Sidebar configuration
with st.sidebar:
    st.header("Connection Settings")
    host = st.text_input("Host", "localhost")
    port = st.number_input("Port", 8005)
    user_id = st.text_input("User ID", "user123")
    
    if st.button("Reset Session"):
        if "ws" in st.session_state:
            try:
                st.session_state.ws.close()
            except Exception:
                pass
            del st.session_state.ws
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        ws_url = f"ws://{host}:{port}/ws/{user_id}"
        
        try:
            # Get or create connection
            if "ws" not in st.session_state:
                st.session_state.ws = websocket.create_connection(ws_url)
            
            ws = st.session_state.ws
            
            # Send message
            try:
                ws.send(prompt)
            except (websocket.WebSocketConnectionClosedException, BrokenPipeError, ConnectionResetError):
                # Reconnect and retry once
                st.warning("Connection lost. Reconnecting...")
                st.session_state.ws = websocket.create_connection(ws_url)
                ws = st.session_state.ws
                ws.send(prompt)

            # Receive loop
            while True:
                chunk = ws.recv()
                if chunk == '\x03':
                    break
                full_response += chunk + "\n\n"
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
            # Clean up broken connection
            if "ws" in st.session_state:
                try:
                    st.session_state.ws.close()
                except Exception:
                    pass
                del st.session_state.ws