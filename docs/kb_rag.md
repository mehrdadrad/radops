# Knowledge Base & RAG Guide

RadOps features a dynamic Knowledge Base (KB) system that implements Retrieval Augmented Generation (RAG). Instead of hardcoding retrieval logic, RadOps automatically generates **Dynamic Tools** for the agent based on your configuration.

This allows the agent to:
1.  **Search** specific document collections (e.g., "Search network configs").
2.  **Filter** results using metadata extracted from filenames (e.g., "Show me configs for site `sjc01`").
3.  **Rerank** results for higher precision using local cross-encoders.

## 1. Vector Store Providers

First, configure the backend database where vectors are stored in `config.yaml`.

```yaml
vector_store:
  providers:
    # Option 1: Weaviate (Recommended)
    weaviate:
      http_host: "localhost"
      http_port: 8080
      grpc_host: "localhost"
      grpc_port: 50051
      # api_key: "..." 

    # Option 2: Chroma (Local)
    chroma:
      path: "./data/chromadb"

    # Option 3: Qdrant
    qdrant:
      url: "http://localhost:6333"
      # api_key: "..."

    # Option 4: Pinecone
    pinecone:
      api_key: "vault:vector#pinecone_key"
      index_name: "radops-index"
      
    # Option 5: Milvus
    milvus:
      connection_args:
        uri: "http://localhost:19530"
```

## 2. Sync Locations (Data Sources)

"Sync Locations" define *what* data is ingested. For every Sync Location defined in `vector_store.profiles`, RadOps creates a corresponding tool for the agent.

### Configuration Structure

```yaml
vector_store:
  profiles:
    - name: "network-docs"        # Used to generate tool name: kb_network_docs
      type: "github"              # Loader type: fs, gdrive, github
      path: "my-org/net-docs"     # Source path
      collection: "network"       # Vector DB collection/class
      sync_interval: 600          # Refresh interval in seconds
      
      # Optional: Customize how the agent sees this tool
      prompt: "Use this tool to search for network topology diagrams and IP plans."
      
      # Optional: Loader specific settings
      loader_config:
        branch: "main"
        file_extensions: [".md", ".txt"]

      # Optional: Metadata extraction rules
      metadata: ...

      # Optional: Retrieval tuning
      retrieval_config: ...
```

### Loader Types

#### File System (`fs`)
Syncs a local directory.
```yaml
type: "fs"
path: "/path/to/local/docs"
```

#### Google Drive (`gdrive`)
Syncs a Google Drive folder by ID. Requires `credentials.json` for a Service Account.
```yaml
type: "gdrive"
path: "1A2B3C4D..." # Folder ID
```

#### GitHub (`github`)
Syncs repositories. Requires `GITHUB_ACCESS_TOKEN` env var or Vault secret.
```yaml
type: "github"
path: "owner/repo"
loader_config:
  branch: "main"
  file_extensions: [".md", ".py"]
```

## 3. Metadata Extraction & Filtering

RadOps can extract metadata from filenames to create **structured filters** for the agent.

**Example:**
If your files are named like `sjc01_router_cisco.txt`, you can define:

```yaml
metadata:
  delimiter: "_"
  structure:
    - name: "site"
      description: "The data center site code (e.g., sjc01)"
    - name: "device_type"
      description: "The type of device (e.g., router, switch)"
    - name: "vendor"
      description: "The hardware vendor"
```

**Result:**
The agent gets a tool `kb_network_docs` with arguments:
*   `query`: string
*   `site`: string (optional)
*   `device_type`: string (optional)
*   `vendor`: string (optional)

The agent can then execute: `kb_network_docs(query="BGP config", site="sjc01")`.

## 4. Retrieval Configuration

You can tune how results are fetched and scored using `retrieval_config`.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `k` | `3` | Number of documents to return. |
| `search_type` | `similarity` | `similarity`, `mmr`, or `similarity_score_threshold`. |
| `score_threshold` | `0.25` | Minimum similarity score (0.0 to 1.0). |
| `rerank` | `{}` | Configuration object for reranking. |
| `rerank.enabled` | `false` | Enable/disable 2-stage reranking. |
| `rerank.provider` | `flashrank` | Rerank provider: `flashrank` or `cohere`. |
| `rerank.model` | (default) | HuggingFace model ID for reranking. |
| `rerank.top_n` | `k` | Number of docs to return after reranking. |

### Search Types

*   **`similarity`**: Standard vector similarity.
*   **`mmr`** (Maximal Marginal Relevance): Optimizes for diversity.
    *   `fetch_k`: Initial pool size (default 20).
    *   `lambda_mult`: Diversity penalty (default 0.5).
*   **`similarity_score_threshold`**: Only returns matches above a certain confidence.

### Reranking

RadOps supports reranking using FlashRank (Local) or Cohere (API) to improve accuracy. It fetches a larger pool of documents first, then re-scores them.

```yaml
retrieval_config:
  k: 5
  search_type: "similarity"
  rerank:
    enabled: true
    provider: "flashrank"
    model: "ms-marco-MiniLM-L-12-v2"
```

## 5. Full Example

Here is a complete configuration for a "Runbooks" knowledge base.

```yaml
vector_store:
  providers:
    weaviate:
      http_host: "localhost"
      http_port: 8080

  profiles:
    - name: "ops-runbooks"
      type: "github"
      path: "acme-corp/ops-runbooks"
      collection: "runbooks"
      sync_interval: 3600
      
      prompt: "Search this for operational procedures, incident response guides, and troubleshooting steps."
      
      loader_config:
        branch: "master"
        file_extensions: [".md"]
      
      metadata:
        delimiter: "-"
        structure:
          - name: "service"
            description: "The service name (e.g., web, db)"
          - name: "severity"
            description: "The incident severity (sev1, sev2)"

      retrieval_config:
        k: 3
        search_type: "mmr"
        rerank:
          enabled: true
```