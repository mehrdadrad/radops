---
name: sample-skill-skeleton
description: "A skeleton template for creating new skills. Use this structure to define operational procedures."
user-invocable: false
metadata:
  { "example": { "requires": { "bins": ["python3"], "env": ["MCP_CALL"] } } }
---

# Skill Title

Provide a comprehensive description of the skill here. What does it do? What systems does it interact with?

## How to Call Tools

Define the standard method for calling tools within this skill.

```bash
# Example:
# python3 $MCP_CALL "tool_name" '{"arg":"value"}'
```

## When to Use

- List scenarios where this skill is applicable.
- Mention any prerequisites.

## Procedure

### Phase 1: Discovery / Information Gathering

#### 1A: Step Description

```bash
# Command to execute
```

**Extraction & Flags:**
- What data to extract?
- What constitutes a warning or critical error?

### Phase 2: Analysis / Execution

#### 2A: Step Description

```bash
# Command to execute
```

## Output

Describe the expected output format (e.g., a report, a JSON blob, a diagram).