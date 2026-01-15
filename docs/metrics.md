# Metrics and Observability Guide

RadOps supports OpenTelemetry for observability, allowing you to collect and export metrics to monitoring systems like Prometheus and Datadog.

## Configuration

OpenTelemetry configuration is managed via the `config/config.yaml` file under the `observability` section.

### Enabling 

You can explicitly enable or disable tracing and metrics using the following flags:

```yaml
observability:
  enable_tracing: true   # Default: false
  enable_metrics: true   # Default: true
```

### Prometheus Exporter

RadOps includes a Prometheus exporter that exposes metrics on a specific address and port.

```yaml
observability:
  prometheus:
    address: 127.0.0.1
    port: 9464
```

- **address**: The interface address to bind the metrics server to (e.g., `127.0.0.1` for localhost or `0.0.0.0` for all interfaces).
- **port**: The port number where metrics will be exposed (default: `9464`).

### Tracing (OTLP)

To send traces to an OTLP-compatible collector (e.g., Jaeger), configure the `tracing_endpoint`:

```yaml
observability:
  enable_tracing: true
  tracing_endpoint: "http://localhost:4317"
```

### Datadog (OTLP)

To send metrics to Datadog (or any OTLP-compatible collector), configure the `metrics_endpoint`:

```yaml
observability:
  metrics_endpoint: "http://localhost:4317"
```



Ensure your Datadog Agent is configured to accept OTLP over gRPC on port 4317.

## Accessing Metrics

Once the application is running, you can access the raw metrics by navigating to the configured address and port in your browser or via `curl`:

```bash
curl http://127.0.0.1:9464/metrics
```

## Integration with Prometheus

To scrape these metrics with a Prometheus server, add a scrape job to your `prometheus.yml` configuration:

```yaml
scrape_configs:
  - job_name: 'radops'
    static_configs:
      - targets: ['localhost:9464']
```

## Available Metrics

The application exposes the following custom metrics via OpenTelemetry:

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `agent.invocations.total` | Counter | Total number of times the agent node is invoked. |
| `agent.llm.duration_seconds` | Histogram | Duration of the LLM call in the agent node. |
| `agent.llm.tokens.total` | Counter | Total number of tokens used by LLM calls. |
| `agent.llm.tokens.cache_read` | Counter | Total number of tokens read from cache by LLM calls. |
| `agent.llm.tokens.cache_creation` | Counter | Total number of tokens created in cache by LLM calls. |
| `agent.tool.duration_seconds` | Histogram | Duration of tool execution. |
| `agent.tool.invocations.total` | Counter | Total number of tool executions. |
| `agent.auditor.score` | Histogram | Quality assurance score assigned by the auditor. |
| `agent.llm.errors` | Counter | Total number of LLM errors. |
| `agent.tool.errors` | Counter | Total number of tool execution errors. |
| `guardrails.blocked.total` | Counter | Total number of requests blocked by guardrails. |
| `agent.memory.operation.duration_seconds` | Histogram | Duration of memory operations (search/add). |
| `agent.memory.items.retrieved` | Counter | Total number of memory items retrieved. |
| `agent.supervisor.plan.size` | Histogram | Number of steps in the generated plan. |

In addition to these custom metrics, the OpenTelemetry instrumentation automatically collects:
- **Runtime Metrics**: Python runtime statistics (GC, memory).

## Supported SaaS Platforms (OTLP)

Since RadOps uses standard OpenTelemetry exporters, it can send data to any platform that supports OTLP (gRPC/HTTP). Common platforms include:

- **Datadog**: Via Datadog Agent OTLP receiver.
- **New Relic**: Direct OTLP ingestion.
- **Dynatrace**: Native OTLP ingestion.
- **Honeycomb**: Native OTLP support.
- **Grafana Cloud**: Via Grafana Alloy or Agent.
- **Splunk Observability Cloud**: Native OTLP support.
- **Lightstep (ServiceNow)**: Native OTLP support.
- **AWS X-Ray / CloudWatch**: Via ADOT Collector.
- **Google Cloud Operations**: Via OpenTelemetry Collector.

## Developer Guide: Adding Traces

To add tracing to your code, you can use the global `telemetry` instance or the standard OpenTelemetry API.

### Using the Telemetry Singleton

```python
from services.telemetry.telemetry import telemetry

def my_function():
    with telemetry.tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("custom.tag", "value")
        # ... code ...
```

### Standard OpenTelemetry Pattern

Since RadOps configures the global tracer provider, you can also use standard patterns:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("my_function")
def my_function():
    pass
```