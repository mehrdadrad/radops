# Metrics and Observability Guide

RadOps supports OpenTelemetry for observability, allowing you to collect and export metrics to monitoring systems like Prometheus and Datadog.

## Configuration

OpenTelemetry configuration is managed via the `config/config.yaml` file under the `opentelemetry` section.

### Prometheus Exporter

RadOps includes a Prometheus exporter that exposes metrics on a specific address and port.

```yaml
opentelemetry:
  prometheus:
    address: 127.0.0.1
    port: 9464
```

- **address**: The interface address to bind the metrics server to (e.g., `127.0.0.1` for localhost or `0.0.0.0` for all interfaces).
- **port**: The port number where metrics will be exposed (default: `9464`).

### Datadog (OTLP)

To send metrics to Datadog (or any OTLP-compatible collector), configure the `endpoint`:

```yaml
opentelemetry:
  endpoint: "http://localhost:4317"
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

The application exposes metrics compatible with the OpenTelemetry standard. While specific metric names depend on the active instrumentation, you can generally expect:

- **Runtime Metrics**: Python runtime statistics (GC, memory).
- **Agent Metrics**: Execution counts and latency for agent workflows.
- **Tool Metrics**: Success/failure rates and duration for external tool calls (e.g., AWS, Network tools).

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