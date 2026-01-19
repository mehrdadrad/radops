# Slack Integration

This module provides a Slack bot interface for the RadOps agent. It uses Slack's **Socket Mode** to receive events, meaning you do not need to expose a public IP or configure firewall ingress rules for Slack webhooks.

## Prerequisites

1. **RadOps Server**: The core server must be running (`python server.py`).
2. **Slack App**: You need permissions to create and install apps in your Slack workspace.

## Configuration

### Required Access (Bot Token Scopes)

Ensure your Slack App has the following scopes enabled under **OAuth & Permissions**:

* **app_mentions:read**: View messages that directly mention @radops in conversations that the app is in
* **calls:write**: Start and manage calls in a workspace
* **channels:history**: View messages and other content in public channels that "radops" has been added to
* **chat:write**: Send messages as @radops
* **chat:write.public**: Send messages to channels @radops isn't a member of
* **im:history**: View messages and other content in direct messages that "radops" has been added to
* **im:read**: View basic information about direct messages that "radops" has been added to
* **im:write**: Start direct messages with people
* **reactions:write**: Add and edit emoji reactions
* **users:read**: View people in a workspace
* **users:read.email**: View email addresses of people in a workspace

### Obtaining Tokens

1. **Create App**: Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app.
2. **App Token (`xapp-...`)**:
   * Go to **Basic Information** > **App-Level Tokens**.
   * Click **Generate Token and Scopes**.
   * Add the `connections:write` scope.
   * Copy the generated token.
3. **Enable Socket Mode**:
   * Go to **Socket Mode** in the sidebar and enable it.
4. **Bot Token (`xoxb-...`)**:
   * Go to **OAuth & Permissions**.
   * Add the scopes listed above.
   * Click **Install to Workspace**.
   * Copy the **Bot User OAuth Token**.

### Slack Configuration

You can configure Slack integration settings in `integrations.yaml`.

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `bot_token` | The Bot User OAuth Token (xoxb-...). | None |
| `app_token` | The App-Level Token (xapp-...). | None |
| `server_url` | WebSocket URL of the RadOps server. | `ws://localhost:8005` |
| `auth_disabled` | Disable authentication (development only). | `false` |
| `auth_method` | Authentication method: `api_key` or `jwt`. | `api_key` |
| `service_api_key` | API key for `api_key` authentication. | None |
| `service_token` | Bearer token for `jwt` authentication. | None |
| `log_level` | Logging level for the Slack adapter. | INFO |
| `inactivity_timeout` | Seconds to wait before closing an inactive WebSocket connection. | 300.0 |

Example `integrations.yaml`:

```yaml
slack:
  bot_token: "xoxb-..."
  app_token: "xapp-..."
  server_url: "ws://localhost:8005"
  auth_method: "api_key"
  service_api_key: "your-secure-api-key"
  log_level: "DEBUG"
  inactivity_timeout: 600
```

**Or using vault references:**

```yaml
slack:
  bot_token: vault:integrations#slack_bot_token
  app_token: vault:integrations#slack_app_token
```


## Dependencies

* `slack-bolt>=1.18.0` - Slack SDK
* `websockets>=10.0` - WebSocket client (version 10.0+ required for `extra_headers` support)

To upgrade websockets:
```bash
pip install --upgrade websockets>=10.0
```