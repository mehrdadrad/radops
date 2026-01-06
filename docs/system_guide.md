# System Features Guide

## Memory Management

RadOps employs a hybrid memory system to maintain context and learn from user interactions.

### Short-Term Memory
Short-term memory retains the context of the current conversation session using Redis. This allows the agent to answer follow-up questions and maintain the flow of dialogue.

**Clearing Short-Term Memory:**
To reset the current conversation context:
*   **Command:** "Clear history" or "Start over".
*   **Action:** This invokes the `delete_conversation_history` tool (if permitted), removing the message history from Redis.

### Long-Term Memory
Long-term memory (powered by Mem0) stores facts, user preferences, and learned information across sessions.

**Managing Long-Term Memory:**
*   The agent automatically updates this memory.
*   To clear specific facts, you can instruct the agent: "Forget that I like Python."

## User Secrets Configuration

Users can securely store credentials for third-party integrations (like Jira and GitHub) directly through the chat. These secrets are stored in HashiCorp Vault.

### Jira
To set up Jira access, provide your username (email) and API token.

**Command:**
```text
Set my Jira secrets: username=user@example.com, token=ATATT3...
```

### GitHub
To set up GitHub access, provide your Personal Access Token.

**Command:**
```text
Set my GitHub secrets: token=ghp_123456...
```

*Note: Ensure you have the `set_user_secrets` permission enabled for your role in `rbac.yaml`.*