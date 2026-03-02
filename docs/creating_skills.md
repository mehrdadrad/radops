# Creating Skills

Skills in RadOps are markdown files that define executable capabilities for agents. They can contain inline code, references to external scripts, or calls to Model Context Protocol (MCP) servers.

## File Structure

A skill file (`SKILL.md` or `*.md`) consists of two parts:
1. **YAML Frontmatter**: Metadata defining the skill's properties.
2. **Markdown Body**: Documentation, input definitions, and inline code blocks.

### Frontmatter Metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique identifier for the skill (e.g., `check-service-status`). |
| `description` | string | **Yes** | A clear description of what the skill does. Used by agents to discover the skill. |
| `type` | string | No | `script` (default), `mcp`, or `inline`. |
| `script` | string | No | Relative path to an external script file (e.g., `scripts/check.py`). |
| `mcp_server` | string | No | Name of the MCP server (required for MCP skills). |
| `mcp_tool` | string | No | Name of the tool on the MCP server (required for MCP skills). |
| `metadata` | json string | No | Additional categorization (e.g., `'{"category":"network"}'`). |

## Execution Modes

The `SkillRunner` executes defined actions in the following order if multiple are present:
1. **MCP Tool**
2. **External Script**
3. **Inline Code Blocks** (executed sequentially from top to bottom)

### 1. Inline Python

Write Python code directly in the markdown file using a `python` code block.

**Handling Inputs:**
- Variables passed to the skill are injected into `locals()`.
- Variables are also mocked into `sys.argv` (e.g., `sys.argv[1]`).

**Example:**
```markdown
---
name: calculate-sum
description: Calculates the sum of two numbers.
---

# Calculate Sum

```python
import sys

# Access via locals() (Recommended)
a = int(locals().get("a", 0))
b = int(locals().get("b", 0))

print(f"The sum is: {a + b}")
```
```

### 2. Inline Bash

Write Bash commands directly in the markdown file using a `bash` or `sh` code block.

**Handling Inputs:**
- Variables are exported as environment variables.

**Example:**
```markdown
---
name: check-disk
description: Checks disk usage for a specific path.
---

# Check Disk

```bash
echo "Checking disk usage for path: $TARGET_PATH"
df -h "$TARGET_PATH"
```
```

### 3. External Script

Reference a script file located relative to the markdown file. Supported extensions: `.py`, `.sh`.

**Handling Inputs:**
- Variables are passed as environment variables to the script.

**Example:**
```markdown
---
name: run-cleanup
description: Runs the cleanup script.
script: ./scripts/cleanup.sh
---

# Cleanup
Documentation for the cleanup script...
```

### 4. MCP Tool

Call a tool exposed by a connected MCP server.

**Example:**
```markdown
---
name: fetch-jira-ticket
description: Gets details of a Jira ticket.
type: mcp
mcp_server: jira
mcp_tool: get_issue
---

# Fetch Jira Ticket
This skill delegates to the Jira MCP server.
```

## Defining Inputs

To help the AI understand what arguments to pass, define an `## Input` section in the markdown body. The system parses this section to generate tool schemas.

```markdown
## Input
- `resource`: The domain name or IP address to query.
- `count`: (Optional) Number of attempts.
```