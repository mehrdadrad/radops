# Creating Workflows

Workflows in RadOps allow you to define complex Standard Operating Procedures (SOPs) as executable Markdown files. Unlike simple Skills, Workflows orchestrate multiple steps involving **Native Tools**, **Agent Skills**, and **MCP Calls**, passing context and variables between them.

## File Location

Workflow definitions are stored in the `workflows/` directory.

## Structure

A workflow file consists of:
1.  **YAML Frontmatter**: Defines metadata and input variables.
2.  **Markdown Body**: Defines the execution steps using numbered lists.

### Frontmatter

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Unique identifier (e.g., `workflow-host-health-check`). |
| `description` | string | **Yes** | Used by the agent to understand when to call this workflow. |
| `metadata` | json string | No | Additional categorization (e.g., `'{"category":"network"}'`). |

### Defining Steps

Steps are defined using a numbered list format with bold titles. Each step contains bullet points for instructions or conditions.

#### Syntax
*   **Step Header**: `1. **Step Name**`
*   **Instruction**: `**Instruction:** <Natural language description>`
    *   The agent interprets this instruction and selects the appropriate tool or skill.
*   **Condition**: `**Condition:** <Logic to skip or alter flow>`
    *   Used to skip steps based on the output of the previous step.

## Example

**File:** `workflows/host_health_check.md`

```markdown
---
name: workflow-host-health-check
description: "Daily System Host Health Check"
metadata: '{"category":"network"}'
---

## Execution

1. **Check DNS**
- **Instruction:** Check the A record for the host
- **Condition:** Skip to "Summarize Findings" if the DNS report does not resolve to an IP address

2. **Check HTTP Status Code**
- **Instruction:** HTTP check for the host

3. **Ping the resolved IP address**
- **Instruction:** Ping the resolved IP address of host

4. **Summarize Findings**
- **Instruction:** Review the outputs generate a summary report and rate the findings for the health check
```