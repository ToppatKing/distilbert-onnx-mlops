from prometheus_client import Counter, Histogram, Gauge

# 1. Latency Tracker (Histogram)
# Tracks the distribution of inference times. 
# We define custom buckets tuned for typical ML inference times (10ms to 2.5s).
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Time spent running ONNX model inference",
    labelnames=["batch_size"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# 2. Throughput Counter
# Tracks the total number of items processed, allowing Grafana to calculate items/second (rate).
INFERENCE_THROUGHPUT = Counter(
    "model_inference_throughput_total",
    "Total number of texts processed by the model",
    labelnames=["batch_size"]
)

# 3. Application State (Gauge)
# Useful for alerting if the application enters a degraded state or the session pool is saturated.
ACTIVE_REQUESTS = Gauge(
    "model_active_requests",
    "Number of inference requests currently being processed"
)

# 4. Model Metadata (Info/Gauge)
# Exposes static information about the model version currently loaded in production.
MODEL_INFO = Gauge(
    "model_metadata",
    "Metadata about the currently loaded model",
    labelnames=["model_id", "quantization", "framework"]
)
