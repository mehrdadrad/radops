# Tools Configuration Guide (`tools.yaml`)

The `tools.yaml` file configures external tools and MCP (Model Context Protocol) servers that the agent can invoke.

## 1. Native Tools

Directly expose Python functions from the application codebase as tools. This is ideal for internal logic or wrappers around local libraries.

### Syntax & Structure

*   **`module`**: The absolute import path to the Python module (e.g., `services.tools...`).
*   **`tools`**: A list of functions within that module to register.
    *   **`function`**: The exact name of the Python function definition.
    *   **`enabled`**: Boolean flag (`true` or `false`) to toggle the tool.

> **How to find tools:** Inspect the `services/tools/` directory in the source code. The `module` parameter corresponds to the Python file path (e.g., `services/tools/network/peeringdb/peeringdb.py` becomes `services.tools.network.peeringdb.peeringdb`), and the `function` parameter matches the specific function name defined inside that file.

### Enabling Local Tools

```yaml
local_tools:
  - module: services.tools.network.peeringdb.peeringdb
    tools:
      - function: network__get_asn_peering_info
        enabled: true
      - function: network__get_peering_exchange_info
        enabled: true
```

### 2. Configuration for built-in integrations.

### Example PeeringDB
Provides access to network peering information.
```yaml
peeringdb:
  api_key: "vault:peeringdb#api_key"
```

### Example GitHub
Allows the agent to interact with GitHub repositories (read issues, check PRs).
```yaml
github:
  server: "https://api.github.com"
  default_org: "my-org"
  default_repo: "network-ops"
  # Token is usually handled via User Secrets or Vault
```


## 2. MCP Servers

Configures Model Context Protocol servers. These run as subprocesses and expose additional tools to the agent.

| Parameter | Description |
| :--- | :--- |
| `command` | The executable to run (e.g., `npx`, `python`, `docker`). |
| `args` | List of arguments for the command. |
| `env` | (Optional) Environment variables for the subprocess. |
| `transport` | The transport mechanism (`stdio` or `streamable_http`). Defaults to `stdio`. |
| `url` | The URL endpoint for HTTP transport. |
| `headers` | (Optional) HTTP headers for HTTP transport. |

### Example: PagerDuty MCP
Allows the agent to manage incidents and on-call schedules.

```yaml
mcp_servers:
  pagerduty: 
    url: "https://mcp.pagerduty.com/mcp"
    transport: "streamable_http"
    headers: 
      Authorization: "vault:tools/pagerduty#token"
```

### Example: Notion MCP
Allows the agent to search and read Notion pages.

```yaml
mcp_servers:
  notion:
    transport: "stdio"
    command: "docker"
    args:
      - "run"
      - "--rm"
      - "-i"
      - "-e"
      - "NOTION_TOKEN"
      - "mcp/notion"
    env:
      NOTION_TOKEN: "vault:tools/notion#token"
```
