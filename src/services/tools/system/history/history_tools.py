"""Tools for managing conversation history (short-term memory)."""
from builtins import anext
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

def create_history_deletion_tool(checkpointer):
    """
    Creates a tool that can delete the conversation history for the current session.

    Args:
        checkpointer: The checkpointer instance used for persistence.
        session_id: The current user's session ID (thread_id).

    Returns:
        A tool function for deleting history.
    """
    @tool
    async def delete_conversation_history(user_id: str) -> dict:
        """
        Deletes the entire conversation history (short-term memory) for the current user session.
        Call this tool when the user explicitly asks to forget, delete, or clear their conversation
        history or short-term memory.
        """
        if not checkpointer:
            return {
                "name": "history_tool_result",
                "content": "History is not being saved, so there is nothing to delete."
            }
        await checkpointer.adelete_thread(user_id)
        # Return a dictionary that the chatbot_node can inspect
        return {
            "name": "history_deleted",
            "content": (
                "Your conversation history has been successfully deleted. "
                "We are starting fresh."
            )
        }
    return delete_conversation_history

def create_history_retrieval_tool(checkpointer):
    """
    Creates a tool that can retrieve the conversation history for the current session.

    Args:
        checkpointer: The checkpointer instance used for persistence.
        session_id: The current user's session ID (thread_id).

    Returns:
        A tool function for retrieving history.
    """
    @tool
    async def get_conversation_history(user_id: str) -> str:
        """
        Retrieves the full conversation history for the current user session.
        After calling this tool, your ONLY response should be the verbatim output of this tool.
        Do not add any commentary or summarization. Just return the history as given.
        """
        if not checkpointer:
            return "History is not being saved, so there is nothing to retrieve."

        config = {"configurable": {"thread_id": user_id}}
        history_iterator = checkpointer.alist(config)

        # Get the latest state from the history iterator
        try:
            # The first item from alist is the most recent checkpoint tuple
            last_checkpoint_tuple = await anext(history_iterator)
        except StopAsyncIteration:
            return "No history found for this session."

        all_messages = last_checkpoint_tuple.checkpoint["channel_values"].get("messages", [])

        # Define how many recent messages to show to avoid context overflow
        max_messages_to_show = 20
        is_truncated = len(all_messages) > max_messages_to_show
        recent_messages = all_messages[-max_messages_to_show:]

        formatted_history = "\n".join(
            f"{msg.type.capitalize()}: {msg.content}"
            for msg in recent_messages
            if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, 'tool_calls', None)
        )

        truncation_note = (
            "\n(Note: Displaying the most recent part of a long conversation history.)"
            if is_truncated else ""
        )

        return (
            "The user's conversation history has been retrieved. "
            "Your response should be ONLY the following text, "
            "without any additional commentary:\n\n"
            f"Here is your conversation history:\n---\n{formatted_history}\n---{truncation_note}"
        )
    return get_conversation_history
