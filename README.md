# RadOps - AI-Powered Network Operations

RadOps is a **Stateful, Multi-Agent DevOps Orchestrator** designed to streamline diagnostics, troubleshooting, and infrastructure management. Unlike standard chatbots, RadOps understands the *lifespan* of information, validates its own work via a QA Auditor, and integrates directly with live infrastructure.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![Tests](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml/badge.svg)](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml)

## 🚀 Key Highlights

* **🛡️ Guardrailed Orchestration**: Uses a Supervisor-Worker architecture with strict sequential logic to prevent execution errors.
* **🧠 3-Tier Cognitive Memory**: Distinguishes between **Working Memory**, **Ephemeral Facts** , and **Core Knowledge** (Permanent Architecture rules).
* **✅ Trust-but-Verify Auditing**: A dedicated **QA Auditor Node** verifies actual tool outputs against the user request to catch hallucinations before they reach you.
* **📂 Declarative RAG & BYODB**: "Bring Your Own Database." Supports top vector databases with zero-code, config-driven knowledge tool generation.
* **🔌 Resilient Connectivity**: Built on the **Model Context Protocol (MCP)** with self-healing clients that survive server restarts.
* **👀 Deep Observability**: Full tracing of Agent Logic, Tool Execution, and LLM Streaming via OpenTelemetry.

## 🛠️ Prerequisites

* **Python 3.10+**
* **Redis** (for conversation checkpointing)
* **HashiCorp Vault** (for secure credential management)
* **Vector Database** (One of: Weaviate, Pinecone, Qdrant, Milvus, ChromaDB)

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/mehrdadrad/radops.git
    cd radops
    ```

2.  Install dependencies (using `uv` for speed):
    ```bash
    uv pip install -e .
    ```

## 🤝 Contribute
We welcome contributions! Please follow these steps:

1.  Fork the project on GitHub.
2.  Create a new feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes.
4.  Push to the branch and open a **Pull Request**.

---
*Built with LangGraph, Mem0, and Passion.*