from prometheus_client import Counter, Gauge, Histogram

detection_total = Counter(
    "detections",
    "Total detections by class",
    ["class_name"],
)

inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "End-to-end inference latency from FastAPI receive to response",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
)

batcher_queue_depth = Gauge(
    "batcher_queue_depth",
    "Current number of frames waiting in AsyncBatcher queue",
)

batcher_batch_size = Histogram(
    "batcher_batch_size",
    "Number of frames per Triton infer call",
    buckets=[1, 2, 4, 8, 16],
)
