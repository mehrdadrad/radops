# Documentation Index

Welcome to the RadOps documentation. Here you will find detailed guides on how to configure, deploy, and use the system.

## Getting Started
*   [Installation & Deployment](quick_start.md) - Prerequisites, installation steps, and running modes (Server vs Console).
*   [Playground](playground.md) - Playground CLI and Streamlit UI to interact with RadOps.
*   [Introduction Video](https://www.youtube.com/watch?v=LVlML1CDI28) - A brief overview of RadOps features (8 min).

## Configuration
*   [Main Configuration](config_guide.md) - Core settings for LLMs, Agents, Memory, and Vector Stores (`config.yaml`).
*   [Tools Configuration](tools_guide.md) - Native Tools Configuring native tools and MCP servers (`tools.yaml`).
*   [RBAC Configuration](rbac_guide.md) - Managing users, roles, and permissions (`rbac.yaml`).

## Features & Integrations
*   [System Features](system_guide.md) - Memory management (Short/Long-term) and user secrets configuration.
*   [Agent Discovery](agent_discovery.md) - Scalable agent routing modes (Prompt vs Discovery).
*   [Integrations & Knowledge Base](integrations_guide.md) - Sync locationsetting up RAG data sources (GitHub, GDrive) and external integrations.
*   [Knowledge Base & RAG](kb_rag.md) - Advanced RAG configuration, vector stores, metadata filtering, and reranking.
*   [Observability](metrics.md) - Configuring OpenTelemetry, Tracing, and Prometheus metrics.
*   [Slack Integration](../src/integrations/slack/README.md) - Configuring the Slack bot and App Manifest for chat interaction.