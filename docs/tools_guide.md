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
mcp:
  servers:
    pagerduty: 
      url: "https://mcp.pagerduty.com/mcp"
      transport: "streamable_http"
      headers: 
        Authorization: "vault:tools/pagerduty#token"
```

### Example: Notion MCP
Allows the agent to search and read Notion pages.

```yaml
mcp:
  servers:
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

### Example: Jenkins MCP
Allows the agent to interact with Jenkins.
The Jenkins MCP plugin must be installed on the target Jenkins server to expose the required endpoints.

```yaml
mcp:
  servers:
    jenkins:
      url: "http://your-jenkins-address/mcp-server/mcp"
      transport: "streamable_http"
      headers:
        Authorization: "vault:tools/jenkins#token"
```

### MCP Client Configuration

The MCP (Model Context Protocol) client supports several configuration parameters to manage connection stability, retries, and timeouts. These can be defined in your agent configuration.

#### Retry and Connection Logic

- **`retry_attempts`** (Default: `3`)
  The number of times the client will immediately try to reconnect if the initial connection fails.

- **`retry_delay`** (Default: `5` seconds)
  The pause duration between immediate retry attempts.

- **`persistent_interval`** (Default: `60` seconds)
  If the connection is lost or initial retries fail, the client enters a persistent background loop. It will attempt to reconnect every `persistent_interval` seconds indefinitely.

- **`health_check_interval`** (Default: `10.0` seconds)
  How frequently the client checks the connection status by listing tools. This helps detect dropped connections (e.g., if the server process died).

#### Timeouts

- **`connect_timeout`** (Default: `10.0` seconds)
  The maximum time allowed for the initial handshake (initialization) and fetching the tool list. If your MCP server is slow to start, increase this value.

- **`execution_timeout`** (Default: `60.0` seconds)
  The maximum time allowed for a tool to run. If a tool call exceeds this duration, it is cancelled and a `TimeoutError` is raised.

#### Example

```yaml
mcp:
  execution_timeout: 120.0  # Allow longer running tools
  retry_attempts: 5         # More aggressive retries
```
