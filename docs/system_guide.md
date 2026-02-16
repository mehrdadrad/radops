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

## Check MCP Health

You can verify the connectivity and status of all configured MCP servers directly through the chat interface. This is useful for debugging connection issues with external tools (e.g., PagerDuty, Notion).

**Command:**
"Check MCP health"


## MCP Server Tools

The `system__list_mcp_server_tools` tool allows you to list available tools from a connected MCP server. This is helpful for discovering what capabilities are exposed by a specific integration.

**Example:**

User: "List the tool names available from the Notion MCP server"

Agent Output:
```text
The system agent has listed the tool names available from the Notion MCP server. The tools are:
- notion__API-get-user: Notion | Retrieve a user
- notion__API-get-users: Notion | List all users
- notion__API-get-self: Notion | Retrieve your token's bot user
- notion__API-post-database-query: Notion | Query a database
- notion__API-post-search: Notion | Search by title
- notion__API-get-block-children: Notion | Retrieve block children
- notion__API-patch-block-children: Notion | Append block children
- notion__API-retrieve-a-block: Notion | Retrieve a block
- notion__API-update-a-block: Notion | Update a block
- notion__API-delete-a-block: Notion | Delete a block
- notion__API-retrieve-a-page: Notion | Retrieve a page
- notion__API-patch-page: Notion | Update page properties
- notion__API-post-page: Notion | Create a page
- notion__API-create-a-database: Notion | Create a database
- notion__API-update-a-database: Notion | Update a database
- notion__API-retrieve-a-database: Notion | Retrieve a database
- notion__API-retrieve-a-page-property: Notion | Retrieve a page property item
- notion__API-retrieve-a-comment: Notion | Retrieve comments
- notion__API-create-a-comment: Notion | Create comment
```