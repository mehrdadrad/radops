# Installation & Deployment Guide

This guide covers the installation requirements and the different operational modes for RadOps.

## Prerequisites

Before running RadOps, ensure the following infrastructure is available:

*   **Python 3.11+**
*   **Redis**: Required for short-term conversation memory.
*   **HashiCorp Vault**: Required for secret management.
*   **Vector Database**: One of Weaviate, Chroma, Qdrant, Milvus or Pinecone (for RAG).

## Quick Start: Infrastructure via Docker

To quickly spin up the required dependencies (Redis, Vault) and an optional Vector DB (e.g., Weaviate), you can use the following `docker-compose.yml`.

1.  **Create `docker-compose.yml`:**

    ```yaml
    version: '3.8'

    services:
      # 1. Redis (Required for Memory)
      redis:
        image: redis:7-alpine
        container_name: radops-redis
        ports:
          - "6379:6379"

      # 2. Vault (Required for Secrets)
      vault:
        image: hashicorp/vault:latest
        container_name: radops-vault
        ports:
          - "8200:8200"
        environment:
          VAULT_DEV_ROOT_TOKEN_ID: 'my-dev-token'
        cap_add:
          - IPC_LOCK

      # 3. Vector DB (Optional - Example: Weaviate)
      weaviate:
        image: semitechnologies/weaviate:1.24.1
        container_name: radops-weaviate
        ports:
          - "8080:8080"
          - "50051:50051"
        environment:
          QUERY_DEFAULTS_LIMIT: 25
          AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
          PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
          DEFAULT_VECTORIZER_MODULE: 'none'
          ENABLE_MODULES: ''
          CLUSTER_HOSTNAME: 'node1'
    ```

2.  **Run the stack:**
    ```bash
    docker-compose up -d
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mehrdadrad/radops.git
    cd radops
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install .
    ```

3.  **Configuration:**
    Copy the example configuration and adjust it to your environment.
    ```bash
    cp config.example.yaml config.yaml
    cp tools.example.yaml tools.yaml
    cp integrations.example.yaml integrations.yaml
    cp rbac.example.yaml rbac.yaml
    ```
    *Refer to the `config_guide.md`, `tools_guide.md`, `integrations_guide.md`, and `rbac_guide.md` for detailed configuration options.*

## Running Modes

RadOps operates in two distinct modes depending on your use case.

### 1. Server Mode (Production)

**Server Mode** is the intended way to run RadOps in a production environment. In this mode, the application:
*   Starts the REST API server to handle incoming requests.
*   Initializes background schedulers for syncing RAG data sources (e.g., GitHub, Google Drive).
*   Logs output to the file specified in `config.yaml` (e.g., `logs/app.log`).

**Command:**
```bash
python server.py
```

**Slack Integration:**

The Slack integration is managed by `src/integrations/slack/slack.py`. This script runs a Slack bot that connects to the RadOps server via WebSockets.

For detailed configuration instructions, required scopes, and token generation, please refer to the [Slack Integration Guide](../src/integrations/slack/README.md).

**Running the Bot:**
Ensure `server.py` is running, then start the Slack bot:
```bash
python src/integrations/slack/slack.py
```

### 2. Console Mode (Debugging & Testing)

**Console Mode** provides an interactive command-line interface (REPL). It is strictly designed for **development, debugging, and testing**.

**Use Console Mode to:**
*   Test new Tool definitions or MCP connections immediately.
*   Debug LLM prompts and responses in real-time.
*   Verify connectivity to infrastructure (Vault, Redis, Vector DB) without starting the full web server.
*   Inspect memory state.

**Command:**
```bash
python console.py
```
