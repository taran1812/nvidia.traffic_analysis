# Observability Design — Week 5

**Goal:** Instrument the FastAPI gateway with Prometheus metrics and build a Grafana dashboard showing request throughput, detection output, inference latency, GPU utilization, and async batcher behavior.

---

## Architecture

```
FastAPI /metrics  ─────────────────────────────────┐
  http_requests_total (instrumentator)               │
  detection_total{class_name}                        ├→ Prometheus (scrape 5s) → Grafana
  inference_duration_seconds (histogram)             │
  batcher_queue_depth (gauge)                        │
  batcher_batch_size (histogram)                     │
Triton :8002/metrics ──────────────────────────────┘
  nv_gpu_utilization
```

---

## File Map

| File | Change |
|---|---|
| `api/metrics.py` | New — 4 custom prometheus_client metrics |
| `api/main.py` | Add instrumentator, mount `/metrics`, observe metrics in detect endpoints |
| `api/batcher.py` | Observe `batcher_queue_depth` and `batcher_batch_size` in `_drain_loop` and `submit` |
| `serving/prometheus.yml` | Add FastAPI scrape job |
| `serving/grafana/dashboards/fastapi.json` | New — 6-panel Grafana dashboard JSON |

---

## Component Details

### `api/metrics.py`

```python
from prometheus_client import Counter, Gauge, Histogram

detection_total = Counter(
    "detection_total",
    "Total detections by class",
    ["class_name"],
)

inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "End-to-end inference latency (FastAPI receive → response)",
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
```

### `api/main.py` changes

- Import `prometheus_fastapi_instrumentator` and mount at startup via lifespan (call `.instrument(app).expose(app)`)
- Add `from api.metrics import detection_total, inference_duration_seconds`
- In `detect_file` and `detect_url`: after `await _batcher.submit(frame)`, observe `inference_duration_seconds.observe(elapsed_ms / 1000)` and for each detection in `dets` call `detection_total.labels(class_name=det.class_name).inc()`

### `api/batcher.py` changes

- Import `batcher_queue_depth`, `batcher_batch_size` from `api.metrics`
- In `submit()`: after `await self._queue.put(...)` call `batcher_queue_depth.inc()`
- In `_drain_loop()`: after draining items, before Triton call, call `batcher_batch_size.observe(len(items))` and after each future resolved call `batcher_queue_depth.dec()`

### `serving/prometheus.yml` changes

Add scrape job for FastAPI:

```yaml
  - job_name: 'fastapi'
    static_configs:
      - targets: ['host.docker.internal:8080']
```

(Uses `host.docker.internal` to reach the FastAPI process running on the host from inside the Prometheus Docker container.)

### `serving/grafana/dashboards/fastapi.json`

6-panel dashboard, all panels use `prometheus` datasource UID `"prometheus"`.

**Panel 1 — Batcher Queue Depth** (line chart)
- Query: `batcher_queue_depth`
- Y-axis: frames

**Panel 2 — Detections by Class** (stacked bar chart)
- Query: `rate(detection_total[1m])` with `class_name` legend
- Y-axis: detections/sec

**Panel 3 — Inference Latency p50 / p95** (line chart, two series)
- Query A: `histogram_quantile(0.5, rate(inference_duration_seconds_bucket[1m]))`
- Query B: `histogram_quantile(0.95, rate(inference_duration_seconds_bucket[1m]))`
- Y-axis: seconds

**Panel 4 — Triton GPU Utilization** (line chart)
- Query: `nv_gpu_utilization`
- Y-axis: percent (0–100)

**Panel 5 — Requests per Second** (line chart)
- Query: `rate(http_requests_total{handler="/detect"}[1m])`
- Y-axis: req/s

**Panel 6 — Batch Size Distribution** (bar chart)
- Query: `rate(batcher_batch_size_bucket[1m])`
- Shows histogram buckets [1,2,4,8,16] — validates async batcher is hitting batch=4 under load

---

## Instrumentation Notes

- `inference_duration_seconds` observed in **seconds** (`elapsed_ms / 1000`) — Prometheus convention; `histogram_quantile` returns seconds
- `batcher_queue_depth` is a Gauge (can go up and down); increment on `submit`, decrement after each future resolved in drain loop
- `batcher_batch_size` observed once per drain iteration (not per frame)
- `prometheus-fastapi-instrumentator` handles HTTP metrics automatically — no manual instrumentation needed for req/s or HTTP latency

---

## prometheus.yml full content

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['triton:8002']

  - job_name: 'fastapi'
    static_configs:
      - targets: ['host.docker.internal:8080']
```

---

## Testing

No new unit tests required — metric instrumentation is fire-and-forget. Integration validation: start the stack, hit `/detect` a few times, curl `http://localhost:8080/metrics` and verify `detection_total`, `inference_duration_seconds_bucket`, `batcher_queue_depth`, `batcher_batch_size_bucket` appear with non-zero values.

---

## Error Handling

- If Prometheus can't reach FastAPI (e.g. not running), scrape fails silently — no impact on the API
- If metric import fails at startup, FastAPI fails to start — caught immediately
- `host.docker.internal` is Docker Desktop's magic hostname for host from container; works on Windows/Mac. On Linux Docker, may need `--add-host=host.docker.internal:host-gateway` in docker-compose.yml

