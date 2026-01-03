# Integrations & Knowledge Base Guide

This guide focuses on configuring data sources for the Knowledge Base (Vector Store) and setting up complex integrations.

## Vector Store Sync Locations

Defined in `config.yaml` under `vector_store.profiles`, these settings determine how documents are ingested into the RAG system.

### Common Configuration

| Parameter | Description |
| :--- | :--- |
| `name` | Identifier for the sync location. |
| `collection` | The collection/namespace in the Vector DB. |
| `sync_interval` | Polling interval in seconds (0 to disable). |
| `metadata` | Rules for extracting metadata from filenames. |

### 1. File System (`fs`)
Syncs files from a local directory.

```yaml
- name: "local-configs"
  type: "fs"
  path: "./data/configs"
  collection: "network_configs"
  sync_interval: 60
```

### 2. Google Drive (`gdrive`)
Syncs files from a Google Drive folder.

**Prerequisites:**
*   `credentials.json` (Service Account or OAuth) must be present in the application root or configured via env vars.

```yaml
- name: "architecture-docs"
  type: "gdrive"
  path: "1A2B3C4D5E6F7G8H9I0J" # Folder ID
  collection: "design_docs"
  sync_interval: 300
```

### 3. GitHub Repository (`github`)
Syncs markdown or code files from a GitHub repository.

```yaml
- name: "ops-runbooks"
  type: "github"
  path: "my-org/runbooks,my-org/scripts" # Comma-separated repos
  collection: "runbooks"
  sync_interval: 600
  loader_config:
    branch: "main"
    file_extensions: [".md", ".py", ".yaml"]
```

## Metadata Extraction

You can automatically tag ingested documents with metadata based on their filenames. This allows the agent to filter searches (e.g., "Show me configs for *sjc01*").

**Mechanism:**
The filename is split by a `delimiter`, and parts are mapped to `structure` fields.

**Example:**
Filename: `sjc01_router01_cisco.txt`

```yaml
metadata:
  delimiter: "_"
  structure:
    - name: "site"       # Maps to "sjc01"
      description: "Data center site code"
    - name: "hostname"   # Maps to "router01"
      description: "Device hostname"
    - name: "vendor"     # Maps to "cisco"
      description: "Hardware vendor"
```

## External Service Setup

### Setting up Google Drive Auth
1.  Go to Google Cloud Console.
2.  Create a Service Account.
3.  Download the JSON key file.
4.  Save as `credentials.json` or set `GOOGLE_APPLICATION_CREDENTIALS` env var.
5.  **Important:** Share the target Google Drive folder with the Service Account email address.

### Setting up GitHub Auth
1.  Create a GitHub Personal Access Token (PAT) with `repo` scope.
2.  Store it in Vault or use the `GITHUB_ACCESS_TOKEN` environment variable.

### Setting up Jira Auth
1.  Use the `set_user_secrets` tool in the chat to store your credentials securely.
2.  **Username**: Your Jira email.
3.  **API Token**: Generate a token from Atlassian Account Settings.

Chat Command:
```
@agent set my jira secrets: username=bob@example.com, token=ATATT3...
```