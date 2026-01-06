# Main Configuration Guide (`config.yaml`)

The `config.yaml` file controls the core behavior of the RadOps application, including AI models, agents, memory persistence, vector database connections and logging.

It serves as the central configuration hub, defining the infrastructure connections and cognitive architecture required for the system to operate. While other configuration files handle specific domains (e.g., `tools.yaml` for capabilities, `rbac.yaml` for permissions), `config.yaml` establishes the foundational environment settings, ensuring the application can connect to necessary services like Redis, Vault, and LLM providers.

## Index

1. [Logging](#logging)
2. [LLM (Large Language Models)](#llm-large-language-models)
3. [Agents](#agents)
4. [Sync Locations (RAG Data Sources)](#sync-locations-rag-data-sources)
5. [Memory & Persistence](#memory--persistence)
6. [Vector Store Providers](#vector-store-providers)
7. [Vault (Secret Management)](#vault-secret-management)
8. [Observability](#observability)

## Logging

Controls the verbosity and output destination of application logs.

| Parameter | Description | Example |
| :--- | :--- | :--- |
| `level` | Logging severity (DEBUG, INFO, WARNING, ERROR). | `"INFO"` |
| `file` | Path to the log file. If omitted, logs go to stdout. | `"/var/log/radops.log"` |
| `retention` | How long to keep log files. | `"1 week"` |
| `rotation` | Size limit before rotating logs. | `"10 MB"` |

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"
  retention: "10 days"
  rotation: "50 MB"
```

## LLM (Large Language Models)

Defines the AI models used by the system. You can define multiple profiles and select a default.

### Supported Providers
*   **OpenAI** (`openai`): Cloud models such as `gpt-4o` and `gpt-4-turbo`.
*   **Anthropic** (`anthropic`): Cloud models such as `claude-3-5-sonnet` and `claude-3-opus`.
*   **DeepSeek** (`deepseek`): DeepSeek API models.
*   **Ollama** (`ollama`): Local models. If used for agents, the model must support tool calling.

| Parameter | Description |
| :--- | :--- |
| `provider` | The model provider (`openai`, `anthropic`, `ollama`, `deepseek`). |
| `model` | The specific model identifier (e.g., `gpt-4o`). |
| `temperature` | Creativity setting (0.0 = deterministic, 1.0 = creative). |
| `api_key` | API key (supports Vault references). |
| `base_url` | Endpoint URL (required for Ollama). |

```yaml
llm:
  default_profile: "openai-main"
  profiles:
    openai-main:
      provider: "openai"
      model: "gpt-4o"
      temperature: 0.0
      api_key: "vault:system#openai_key"
    
    ollama-local:
      provider: "ollama"
      model: "llama3"
      base_url: "http://localhost:11434"

    deepseek-main:
      provider: "deepseek"
      model: "deepseek-coder"
      api_key: "vault:system#deepseek_key"
```

## Agents

Configures the specialized agents that the Supervisor delegates tasks to.

### Adding a Custom Agent

To create a new agent, add an entry under `agent.profiles`.

| Parameter | Description |
| :--- | :--- |
| `description` | Used by the Supervisor to route requests. Be specific about what the agent can and cannot do. |
| `llm_profile` | The ID of the LLM profile to use (defined in the `llm` section). |
| `manifest_llm_profile` | (Optional) A specific profile for generating the agent's capability manifest at startup. |
| `system_prompt_file` | Path to the text file containing the agent's instructions. |
| `allow_tools` | List of regex patterns matching the tool names this agent can access. |

```yaml
agent:
  profiles:
    # Example: A new Network Specialist
    network_specialist:
      description: "Specialist for network diagnostics and configuration."
      llm_profile: "openai-main"
      system_prompt_file: "config/prompts/network_specialist.txt"
      allow_tools:
        - system__.*      # Required for submitting work
        - network__.*     # Custom tools
        - kb_network      # Knowledge base tool
```

### Core System Agents

You can also configure the built-in system agents.

| Parameter | Description |
| :--- | :--- |
| `threshold` | (Auditor) Confidence score (0.0-1.0) required to approve an action. |

```yaml
agent:
  supervisor:
    llm_profile: "openai-main"
  auditor:
    enabled: true
    llm_profile: "openai-main"
    threshold: 0.8
```

## Sync Locations (RAG Data Sources)

Defined under `vector_store.profiles`, these settings control which data sources are ingested into the Knowledge Base.

### Supported Loaders
*   **File System** (`fs`): Local directories.
*   **Google Drive** (`gdrive`): Remote Google Drive folders.
*   **GitHub** (`github`): GitHub repositories (code or docs).

| Parameter | Description |
| :--- | :--- |
| `name` | Unique identifier for the sync job. |
| `type` | The loader type (`fs`, `gdrive`, `github`). |
| `path` | The source location (path, ID, or repo slug). |
| `collection` | The destination collection in the Vector DB. |
| `sync_interval` | Polling interval in seconds. |
| `loader_config` | (Optional) Loader-specific settings (e.g., branch, extensions). |

```yaml
vector_store:
  profiles:
    - name: "ops-runbooks"
      type: "github"
      path: "my-org/runbooks"
      collection: "runbooks"
      sync_interval: 600
      loader_config:
        branch: "main"
        file_extensions: [".md", ".py", ".yaml"]
```

> **Note:** For detailed setup instructions (e.g., Google Drive credentials), refer to the **Integrations Guide**.

## Memory & Persistence

Configures Short-term (Redis) and Long-term (Mem0) memory.

### Redis (Short-term)
Stores active conversation history.

```yaml
memory:
  redis:
    endpoint: "redis://localhost:6379"
    ttl:
      time_minutes: 60
      refresh_on_read: true
```

### Summarization
Configures how the agent manages context window limits by summarizing older parts of the conversation.

| Parameter | Description |
| :--- | :--- |
| `keep_message` | The number of most recent messages to keep in their raw format. Older messages are summarized or pruned. |
| `token_threshold` | The token count threshold that triggers the summarization process. |
| `llm_profile` | The LLM profile used to generate the summary. If left empty, old messages are pruned without summarization. |

```yaml
memory:
  summarization:
    keep_message: 40
    token_threshold: 2000
    llm_profile: "openai-summary"
```

### Mem0 (Long-term)
Stores user facts and preferences across sessions.

```yaml
mem0:
  llm_profile: "openai-main"
  embedding_profile: "openai-embedding"
  vector_store:
    provider: "weaviate"
    config:
      collection_name: "Mem0_Memory"
      cluster_url: "http://localhost:8080"
  excluded_tools:
    - "set_user_secrets"
```

## Vector Store Providers

Configures the connection details for the Vector Database used for RAG (Retrieval Augmented Generation).

**Note:** For configuring *what* data to sync (Sync Locations), please refer to the Integrations Guide.

```yaml
vector_store:
  providers:
    weaviate:
      http_host: "localhost"
      http_port: 8080
      grpc_host: "localhost"
      grpc_port: 50051
    
    chroma:
      path: "./data/chromadb"
      
    pinecone:
      api_key: "vault:vector#pinecone_key"
      index_name: "radops-index"
```

## Vault (Secret Management)

Configures the connection to HashiCorp Vault for secure secret retrieval.

```yaml
vault:
  url: "http://localhost:8200"
  token: "root-token" # Recommended: Use VAULT_TOKEN env var instead
  mount_point: "secret"
```

### Using Secrets
Reference secrets in any config file using the syntax: `vault:<path>#<key>`.

Example: `api_key: "vault:system/openai#key"`
