# RadOps - AI-powered Network Operations

A sophisticated AI-powered network operations assistant designed to help with network diagnostics, troubleshooting, and infrastructure management. Built with LangGraph and LangChain, it leverages Large Language Models (LLMs) to provide intelligent responses and execute tools securely.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![Tests](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml/badge.svg)](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml)

## Features

*   **Intelligent Agent**: Uses LLMs (OpenAI, Ollama, DeepSeek) to understand and process user requests.
*   **Network Tools**:
    *   PeeringDB integration for ASN and exchange information.
    *   Network diagnostics (Ping, Traceroute, etc.).
    *   Looking Glass integration.
*   **Tools**:    
    *   **Jira**: Create and search issues.
    *   **GitHub**: Manage issues and pull requests.
*   **Integrations**:
    *   **Slack**: Receive requests and interact directly via Slack.
    *   **CI/CD**: Easy integration via WebSockets for automated pipelines.    
*   **Knowledge Base (RAG)**:
    *   Ingests documentation from File System and Google Drive.
    *   Supports Vector Stores like Weaviate and Chroma.
*   **Memory & Context**:
    *   **Short-term**: Redis-backed conversation history.
    *   **Long-term**: Mem0 integration for personalized user memory.
*   **Security**:
    *   **RBAC**: Role-Based Access Control for tool execution.
    *   **Vault**: HashiCorp Vault integration for secure secret management.

## Prerequisites

*   Python 3.10+
*   Redis (for conversation memory)
*   HashiCorp Vault (for secret management)
*   Weaviate or Chroma (for vector store)

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application is highly configurable via YAML files in the `config/` directory.

*   **`config.yaml`**: Configure LLM providers, Vector Store settings, Memory backends, and Vault connection.
*   **`rbac.yaml`**: Define users, roles, and allowed tools per role.
*   **`tools.yaml`**: Configure credentials and settings for external tools.

For a detailed guide on configuration, please refer to docs/configuration_guide.md.

## Usage

To start the application:

```bash
python src/console.py
```

You will be prompted to enter a username (which maps to roles in `rbac.yaml`) to start the session.
