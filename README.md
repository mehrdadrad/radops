# RadOps - AI-Powered Operations

RadOps is an AI-powered, multi-agent platform that automates DevOps workflows 
with human-level reasoning. Unlike traditional chatbots, RadOps remembers 
context, validates its own work, and executes complex multi-step operations 
across your entire infrastructure—autonomously.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![Tests](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml/badge.svg)](https://github.com/mehrdadrad/radops/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mehrdadrad/radops/branch/main/graph/badge.svg)](https://codecov.io/gh/marshadrad/radops)

![RadOps Architecture](/assets/radops_arch_diagram.png)

[▶️ Watch the Introduction Video (8 min)](https://www.youtube.com/watch?v=LVlML1CDI28)

## 🚀 Key Highlights

* **🛡️ Guardrailed Orchestration**: Uses a Supervisor-Worker architecture with strict sequential logic to prevent execution errors.
* **🧠 3-Tier Cognitive Memory**: Distinguishes between **Working Memory**, **Ephemeral Facts** , and **Core Knowledge** (Permanent Architecture rules).
* **🔎 Scalable Agent Discovery**: Supports **Prompt Mode** for small teams (<15 agents) and **Discovery Mode** for unlimited scaling via vector-based agent lookup.
* **🤖 Config-Driven Specialists**: Instantly spin up specialized agents (e.g., Network, Security) by defining personas and toolsets in YAML — no new code required.
* **👨‍💻 Human-in-the-Loop**: Seamlessly pause workflows for user approval or input before executing sensitive actions.
* **⚡ Agent Skills**: Rapidly extend agent capabilities by defining executable skills (Python, Bash, or MCP calls) directly in Markdown files.
* **📜 Unified Workflows**: Define complex Standard Operating Procedures (SOPs) in Markdown that orchestrate **Tools**, **Skills**, and **MCP** calls with variable injection.
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

## 💡 Example Outputs

### Workflow
```
User: run host health check workflow for www.ippacket.org

================================== Ai Message ==================================

You requested to run a host health check workflow for www.ippacket.org. The plan is to perform the following steps:

1. Check the A record (DNS) for www.ippacket.org.
2. Perform an HTTP check for www.ippacket.org.
3. Ping the resolved IP address of www.ippacket.org from multiple nodes.
4. Review the outputs from the DNS, HTTP, and ping checks, generate a summary report.

I will start with the first step: checking the A record for www.ippacket.org.

================================== Ai Message ==================================

The summary report for the host health check of www.ippacket.org is ready:

- DNS Check:
  - Resolved IP: 74.208.236.26
  - TTL: 300 seconds
  - Successful resolution from all nodes.

- HTTP Check:
  - HTTP status: 200 OK from two nodes
  - One node experienced a connect timeout
  - Response times: 0.243s, 0.404s
  - IP confirmed: 74.208.236.26

- Ping Check:
  - Ping latency times (seconds):
    - bg1.node.check-host.net: ~0.140
    - rs1.node.check-host.net: ~0.142
    - ua3.node.check-host.net: ~0.141
  - All pings successful with consistent low latency.

Conclusion:
www.ippacket.org is reachable and responsive with stable DNS resolution, mostly successful HTTP service, and low-latency ping responses.
```

### Single task:

```
User: get asn 701 peering info

================================== Ai Message ==================================

The peering information for ASN 701 has been retrieved:
- Organization: Verizon (UUNET, MCI, Verizon Business)
- ASN: 701
- Website: https://www.verizon.com/business/
- Traffic: 1-5Tbps
- Prefixes IPv4: 100000
- Prefixes IPv6: 25000
- Scope: Global
- IPv6 Support: Yes
- Multicast Support: No
- IX Count: 1
- Facility Count: 19
- Peering Policy: Restrictive, US locations required, contracts required
- RPKI Validation: Implemented in 4Q 2022
- Notes: AS 701 will implement RPKI validation and reject invalid prefixes from peers and customers. Please review your registered ROAs to reduce invalid prefixes.

There are no further steps in your request. If you need additional information or actions, please let me know.

```

### Conditional task:

```
User: check HTTP for www.ippacket.org then if the return code was not 200 then ping the host

================================== Ai Message ==================================

You requested to check the HTTP status for www.ippacket.org and if the return code is not 200, then ping the host. I have created the following plan:

1. Perform an HTTP check for www.ippacket.org.
2. If the HTTP return code is not 200, then ping the host www.ippacket.org.

I will start by instructing the network agent to perform the HTTP check for www.ippacket.org.

================================== Ai Message ==================================

The HTTP check for www.ippacket.org was successful with a return code of 200 from multiple nodes:
- Node ca1: 200 OK
- Node ch1: 200 OK
- Node ua1: 200 OK
Since the return code is 200, the next step to ping the host is not required as per the plan.

================================== Ai Message ==================================

The HTTP check for www.ippacket.org returned code 200 from all nodes, so the conditional step to ping the host is skipped as it is not needed.

```

### Multi steps task

```
User: get stopped AWS instances then check jenkins piplines running jobs 

================================== Ai Message ==================================

I understand you want to get the list of stopped AWS instances and then check the running jobs in Jenkins pipelines. Here is the plan I have created to fulfill your request:

1. Retrieve the stopped AWS instances using the cloud agent.
2. Check the running jobs in Jenkins pipelines using the support agent.

I will start by retrieving the stopped AWS instances now.

================================== Ai Message ==================================

The stopped AWS instances have been retrieved:
- Instance ID: i-0ddec155d2df9c8eb
- Name: my-instance2
- Instance Type: t3.micro
- State: stopped
- Private IP: 172.31.14.8
- VPC ID: vpc-6b83ad0e
- Subnet ID: subnet-f0f39ba9
- Launch Time: 2025-12-27 17:29:18+00:00

Now, I will proceed to check the running jobs in Jenkins pipelines.

================================== Ai Message ==================================

The Jenkins pipelines running jobs have been checked. There are no jobs currently building or in progress. The latest builds for the pipeline "pipeline 111" are all completed successfully.

```

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