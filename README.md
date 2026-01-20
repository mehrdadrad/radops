# RadOps - AI-Powered Operations

RadOps is an AI-powered, multi-agent platform that automates DevOps workflows 
with human-level reasoning. Unlike traditional chatbots, RadOps remembers 
context, validates its own work, and executes complex multi-step operations 
across your entire infrastructure—autonomously.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![Tests](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml/badge.svg)](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml)

![RadOps Architecture](/assets/radops_arch_diagram.png)

## 🚀 Key Highlights

* **🛡️ Guardrailed Orchestration**: Uses a Supervisor-Worker architecture with strict sequential logic to prevent execution errors.
* **🧠 3-Tier Cognitive Memory**: Distinguishes between **Working Memory**, **Ephemeral Facts** , and **Core Knowledge** (Permanent Architecture rules).
* **🤖 Config-Driven Specialists**: Instantly spin up specialized agents (e.g., Network, Security) by defining personas and toolsets in YAML — no new code required.
* **👨‍💻 Human-in-the-Loop**: Seamlessly pause workflows for user approval or input before executing sensitive actions.
* **🔄 Multi-Step Workflows**: Automatically decomposes complex requests into logical steps, executing them sequentially with state tracking and plan enforcement.
* **✅ Trust-but-Verify Auditing**: A dedicated **QA Auditor Node** verifies actual tool outputs against the user request to catch hallucinations before they reach you.
* **📂 Declarative RAG & BYODB**: "Bring Your Own Database." Supports top vector databases with zero-code, config-driven knowledge tool generation.
* **🔌 Resilient Connectivity**: Built on the **Model Context Protocol (MCP)** with self-healing clients that survive server restarts.
* **👀 Deep Observability**: Full tracing of Agent Logic, Tool Execution, and LLM Streaming via OpenTelemetry.

## 🧠 Supported Providers

### LLM Providers
* **OpenAI** (`openai`): Cloud models such as `gpt-5` and `gpt-5-nano`.
* **Anthropic** (`anthropic`): Cloud models such as `claude-4-5-sonnet` and `claude-4-5-opus`.
* **DeepSeek** (`deepseek`): DeepSeek API models.
* **Azure OpenAI** (`azure`): Azure hosted OpenAI models.
* **Google** (`google`): Google Gemini models such as `gemini-3-pro-preview`.
* **Groq** (`groq`): Groq Cloud models.
* **Mistral** (`mistral`): Mistral AI models.
* **AWS Bedrock** (`bedrock`): AWS managed models.
* **Ollama** (`ollama`): Local models. If used for agents, the model must support tool calling.

### Vector Databases
* **Weaviate** Hybrid search, GraphQL API, multi-tenancy
* **Qdrant** High performance (Rust), advanced filtering
* **Pinecone** Managed cloud, serverless, auto-scaling
* **Milvus** Open source, horizontal scaling, GPU support
* **Chroma** Lightweight, embedded, perfect for dev/test

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

## 📚 Documentation

For detailed guides on configuration, deployment, and features, please refer to the [documentation](docs/README.md).


## 🤝 Contribute
We welcome contributions! Please follow these steps:

1.  Fork the project on GitHub.
2.  Create a new feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes.
4.  Push to the branch and open a **Pull Request**.

---
*Built with LangGraph, Mem0, Top Vector Databases, and Passion.*