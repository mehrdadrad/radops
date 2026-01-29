import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

class AdaptiveLearningEngine:
    """
    Adaptive Learning Engine for RadOps.
    
    This class is responsible for logging successful interaction sequences 
    to a dataset file (JSONL) which can be used to fine-tune models later.
    """
    
    def __init__(self, dataset_path: str = "data/fine_tuning_dataset.jsonl"):
        self.dataset_path = dataset_path

    def log_interaction(self, messages: List[BaseMessage], summary: str = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Logs the interaction history to the dataset file.
        
        Args:
            messages: The list of messages from the conversation state.
            summary: Optional summary of previous conversation context.
            metadata: Optional metadata (e.g., user_id, success_flag).
        """
        if not messages and not summary:
            return

        # Identify all available tool outputs to prevent orphaned tool calls
        available_tool_ids = {
            msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
        }

        # Step 1: Convert to intermediate format and filter orphaned tool calls
        temp_messages = []
        
        if summary:
            temp_messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})

        for msg in messages:
            role = "user"
            content = msg.content
            tool_calls = None
            tool_call_id = None
            
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                # Filter out Auditor completion messages to keep conversation flow clean
                if msg.content and "QA Passed" in str(msg.content):
                    continue

                role = "assistant"
                if msg.tool_calls:
                    # Filter tool calls that have a corresponding tool output
                    valid_tool_calls = [
                        tc for tc in msg.tool_calls
                        if tc["id"] in available_tool_ids
                    ]
                    
                    if valid_tool_calls:
                        tool_calls = [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["args"])
                                }
                            }
                            for tc in valid_tool_calls
                        ]
            elif isinstance(msg, ToolMessage):
                role = "tool"
                tool_call_id = msg.tool_call_id

            # Skip empty messages (no content and no tool calls)
            if not content and not tool_calls:
                continue

            message_dict = {"role": role}
            if content:
                message_dict["content"] = content
            if tool_calls:
                message_dict["tool_calls"] = tool_calls
            if tool_call_id:
                message_dict["tool_call_id"] = tool_call_id
            
            temp_messages.append(message_dict)

        # Step 2: Clean up conversation structure
        final_messages = []
        
        # Preserve system message if present
        if temp_messages and temp_messages[0]["role"] == "system":
            final_messages.append(temp_messages[0])
            
        # Find first user message to start the conversation flow
        messages_to_process = []
        found_first_user = False
        for msg in temp_messages:
            if not found_first_user and msg["role"] == "user":
                found_first_user = True
            
            if found_first_user:
                messages_to_process.append(msg)

        if not messages_to_process:
            return

        skip_next = False

        for i, msg in enumerate(messages_to_process):
            if skip_next:
                skip_next = False
                continue

            if (msg["role"] == "user" and 
                "COMMAND FROM SUPERVISOR" in msg.get("content", "") and 
                "Do NOT reply with text" in msg.get("content", "")):
                
                # Look ahead to the next message
                if i + 1 < len(messages_to_process):
                    next_msg = messages_to_process[i+1]
                    # If next is Assistant responding with text ONLY (no tools), it's a broken chain
                    if next_msg["role"] == "assistant" and "tool_calls" not in next_msg:
                        skip_next = True
                        continue
            
            final_messages.append(msg)

        if not final_messages:
            return

        entry = {"messages": final_messages}
        if metadata:
            entry["metadata"] = metadata
        
        # Append to JSONL file
        try:
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"Successfully logged interaction to {self.dataset_path}")
        except Exception as e:
            logger.error(f"Failed to log interaction for fine-tuning: {e}")

    def export_fine_tuning_data(self) -> str:
        """
        Returns the path to the current fine-tuning dataset.
        """
        return self.dataset_path