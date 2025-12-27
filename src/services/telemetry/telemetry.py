import logging
from typing import Optional, Dict, Any, List, Callable, Iterable
from threading import Lock
 
from opentelemetry import trace, metrics, _logs
from opentelemetry.trace import NoOpTracerProvider
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry._logs import NoOpLoggerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from wsgiref.simple_server import WSGIServer

# To export to an OTLP collector (e.g., Jaeger, Prometheus, Loki) you would use the following exporters.
# The PrometheusMetricReader is used to expose a /metrics endpoint directly from the app.
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader, start_http_server

from config.config import settings
from services.telemetry.metrics import register_default_metrics

logger = logging.getLogger(__name__)

class Telemetry:
    """
    A singleton class to manage OpenTelemetry setup, and metric creation/updates.
    """
    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                # Another check for thread safety
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, 
                 otel_endpoint: Optional[str] = None,
                 enable_tracing: bool = False,
                 enable_metrics: bool = True,
                 enable_logging: bool = False,
                 server: WSGIServer = None
                 ):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return

            self.service_name = "radops"
            self.otel_endpoint = otel_endpoint
            self.enable_tracing = enable_tracing
            self.enable_metrics = enable_metrics
            self.enable_logging = enable_logging

            self._setup_opentelemetry()

            # For instrument management
            # These will get a NoOp provider if the corresponding signal is disabled
            self.tracer = trace.get_tracer(self.service_name) 
            self.meter = metrics.get_meter(self.service_name) 
            
            self._metrics: Dict[str, Any] = {}
            register_default_metrics(self)
            
            self.__class__._initialized = True
            logger.info(f"Telemetry singleton initialized for service '{self.service_name}'.")

    def _setup_opentelemetry(self):
        """Configures OpenTelemetry for metrics, logging, and tracing."""
        resource = Resource(attributes={SERVICE_NAME: self.service_name})
    
        self.server = None
        
        # --- Tracing Setup ---
        if self.enable_tracing:
            tracer_provider = TracerProvider(resource=resource)
            if self.otel_endpoint:
                print(f"Using OTLP endpoint for tracing: {self.otel_endpoint}")
                span_exporter = OTLPSpanExporter(endpoint=self.otel_endpoint, insecure=True)
            else:
                span_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            trace.set_tracer_provider(tracer_provider)
        else:
            logger.info("Tracing is disabled.")
            trace.set_tracer_provider(NoOpTracerProvider())

        # --- Metrics Setup ---
        if self.enable_metrics:
            address = settings.opentelemetry.get("prometheus", {}).get("address", "localhost")
            port = settings.opentelemetry.get("prometheus", {}).get("port", 9464)
            logger.info(f"Starting Prometheus server on {address}:{port}")
            self.server, _ = start_http_server(port=port, addr=address)
            metric_reader = PrometheusMetricReader()
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
        else:
            print("Metrics are disabled.")
            metrics.set_meter_provider(NoOpMeterProvider())

        # --- Logging Setup ---
        if self.enable_logging:
            logger_provider = LoggerProvider(resource=resource)
            # Logs will always go to the console.
            log_exporter = ConsoleLogExporter()
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
            _logs.set_logger_provider(logger_provider)
            
            # Integrate with Python's standard logging
            handler = LoggingHandler(level=logging.getLogger().level, logger_provider=logger_provider)
            logging.getLogger().addHandler(handler)
        else:
            logger.info("Logging is disabled.")
            _logs.set_logger_provider(NoOpLoggerProvider())

    def shutdown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def register_counter(self, name: str, unit: str = "", description: str = ""):
        """Registers a new counter metric."""
        if name in self._metrics:
            logger.warning(f"Metric '{name}' is already registered.")
            return
        self._metrics[name] = self.meter.create_counter(name, unit, description)
        logger.info(f"Registered counter: {name}")

    def update_counter(self, name: str, value: int = 1, attributes: Optional[Dict[str, str]] = None):
        """Updates a counter metric by a given value."""
        if name not in self._metrics:
            logger.error(f"Counter '{name}' not registered. Please register it first.")
            return
        self._metrics[name].add(value, attributes)

    def register_histogram(self, name: str, unit: str = "", description: str = ""):
        """Registers a new histogram metric."""
        if name in self._metrics:
            logger.warning(f"Metric '{name}' is already registered.")
            return
        self._metrics[name] = self.meter.create_histogram(name, unit, description)
        logger.info(f"Registered histogram: {name}")

    def update_histogram(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None):
        """Records a value in a histogram metric."""
        if name not in self._metrics:
            logger.error(f"Histogram '{name}' not registered. Please register it first.")
            return
        self._metrics[name].record(value, attributes)

    def register_up_down_counter(self, name: str, unit: str = "", description: str = ""):
        """Registers a new up-down-counter metric."""
        if name in self._metrics:
            logger.warning(f"Metric '{name}' is already registered.")
            return
        self._metrics[name] = self.meter.create_up_down_counter(name, unit, description)
        logger.info(f"Registered up-down-counter: {name}")

    def update_up_down_counter(self, name: str, value: int, attributes: Optional[Dict[str, str]] = None):
        """Updates an up-down-counter metric by a given value."""
        if name not in self._metrics:
            logger.error(f"Up-down-counter '{name}' not registered. Please register it first.")
            return
        self._metrics[name].add(value, attributes)

    def register_observable_gauge(
        self,
        name: str,
        callbacks: List[Callable[[Any], Iterable[Observation]]],
        unit: str = "",
        description: str = ""
    ):
        """Registers a new observable gauge metric that reports values via callbacks."""
        if name in self._metrics:
            logger.warning(f"Metric '{name}' is already registered.")
            return
        # The instrument is stored for bookkeeping, but not used for updates.
        # The SDK invokes the callbacks automatically.
        self._metrics[name] = self.meter.create_observable_gauge(name, callbacks, unit, description)
        logger.info(f"Registered observable gauge: {name}")

# Singleton instance
telemetry = Telemetry()
       
