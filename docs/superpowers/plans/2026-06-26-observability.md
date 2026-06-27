# Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the FastAPI gateway with Prometheus metrics and add a 6-panel Grafana dashboard showing request throughput, detection output, inference latency, GPU utilization, and async batcher behavior.

**Architecture:** `api/metrics.py` defines 4 prometheus_client objects (Counter, Gauge, 2× Histogram). `api/main.py` mounts `prometheus-fastapi-instrumentator` for HTTP metrics and exposes `/metrics`, then observes the custom metrics inside each detect endpoint. `api/batcher.py` observes queue depth and batch size. Prometheus scrapes both Triton (existing) and FastAPI (new) every 5s. A new Grafana dashboard JSON is provisioned automatically.

**Tech Stack:** `prometheus_client`, `prometheus-fastapi-instrumentator`, Grafana JSON provisioning, existing Prometheus + Grafana Docker Compose stack.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `api/metrics.py` | **Create** | 4 prometheus_client metric objects |
| `api/requirements.txt` | **Modify** | Add `prometheus-fastapi-instrumentator>=0.9.1` |
| `api/main.py` | **Modify** | Mount instrumentator, expose `/metrics`, observe metrics in detect endpoints |
| `api/batcher.py` | **Modify** | Observe `batcher_queue_depth` and `batcher_batch_size` |
| `api/tests/test_main.py` | **Modify** | Add test for `/metrics` endpoint |
| `serving/prometheus.yml` | **Modify** | Add FastAPI scrape job |
| `serving/grafana/dashboards/fastapi.json` | **Create** | 6-panel Grafana dashboard |

---

### Task 1: Metrics module + /metrics endpoint

**Files:**
- Create: `api/metrics.py`
- Modify: `api/requirements.txt`
- Modify: `api/main.py`
- Modify: `api/tests/test_main.py`

**Context:** `api/main.py` already has a `lifespan` context manager and `app = FastAPI(...)`. The `client` fixture in `test_main.py` patches `AsyncBatcher` and `TritonClient`.

- [ ] **Step 1: Add dependency to requirements.txt**

Open `api/requirements.txt` and add one line:
```
prometheus-fastapi-instrumentator>=0.9.1
```

Install it:
```bash
cd /home/sai_taran/nvidia-traffic-analytics && source venv/bin/activate
pip install prometheus-fastapi-instrumentator>=0.9.1
```

- [ ] **Step 2: Create `api/metrics.py`**

```python
from prometheus_client import Counter, Gauge, Histogram

detection_total = Counter(
    "detection_total",
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
```

- [ ] **Step 3: Write failing test**

Add to `api/tests/test_main.py` (inside the file, after existing tests):

```python
def test_metrics_endpoint_exists(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
```

- [ ] **Step 4: Run test — verify RED**

```bash
cd /home/sai_taran/nvidia-traffic-analytics && source venv/bin/activate
python -m pytest api/tests/test_main.py::test_metrics_endpoint_exists -v
```

Expected: FAIL — 404 Not Found (no `/metrics` route yet).

- [ ] **Step 5: Mount instrumentator in `api/main.py`**

Add after the existing imports at the top of `api/main.py`:
```python
from prometheus_fastapi_instrumentator import Instrumentator
```

Add after `app = FastAPI(title="Traffic Analytics API", lifespan=lifespan)`:
```python
Instrumentator().instrument(app).expose(app)
```

- [ ] **Step 6: Run test — verify GREEN**

```bash
python -m pytest api/tests/test_main.py -v
```

Expected: all tests pass (including the new `test_metrics_endpoint_exists`). If any previous tests break, the instrumentator may interfere with the mock setup — fix by adjusting the `client` fixture to reset the metric registry.

- [ ] **Step 7: Commit**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git add api/metrics.py api/requirements.txt api/main.py api/tests/test_main.py
git commit -m "feat: add prometheus metrics module and /metrics endpoint"
```

---

### Task 2: Observe metrics in detect endpoints

**Files:**
- Modify: `api/main.py`

**Context:** `detect_file` and `detect_url` in `api/main.py` both call `await _batcher.submit(frame)` which returns `(dets, elapsed_ms)`. `dets` is `list[Detection]`, each with a `.class_name` str. `elapsed_ms` is float in milliseconds — observe as `elapsed_ms / 1000` seconds.

- [ ] **Step 1: Import custom metrics in `api/main.py`**

Add to the imports block:
```python
from api.metrics import detection_total, inference_duration_seconds
```

- [ ] **Step 2: Observe metrics in `detect_file`**

Current `detect_file` body after submit:
```python
    dets, elapsed_ms = await _batcher.submit(frame)
    return DetectResponse(
        detections=dets,
        inference_time_ms=elapsed_ms,
        model="yolov8n",
        image_size=[h, w],
    )
```

Replace with:
```python
    dets, elapsed_ms = await _batcher.submit(frame)
    inference_duration_seconds.observe(elapsed_ms / 1000)
    for det in dets:
        detection_total.labels(class_name=det.class_name).inc()
    return DetectResponse(
        detections=dets,
        inference_time_ms=elapsed_ms,
        model="yolov8n",
        image_size=[h, w],
    )
```

- [ ] **Step 3: Observe metrics in `detect_url`**

Current `detect_url` body after submit:
```python
    dets, elapsed_ms = await _batcher.submit(frame)
    return DetectResponse(
        detections=dets,
        inference_time_ms=elapsed_ms,
        model="yolov8n",
        image_size=[h, w],
    )
```

Replace with:
```python
    dets, elapsed_ms = await _batcher.submit(frame)
    inference_duration_seconds.observe(elapsed_ms / 1000)
    for det in dets:
        detection_total.labels(class_name=det.class_name).inc()
    return DetectResponse(
        detections=dets,
        inference_time_ms=elapsed_ms,
        model="yolov8n",
        image_size=[h, w],
    )
```

- [ ] **Step 4: Run full test suite**

```bash
cd /home/sai_taran/nvidia-traffic-analytics && source venv/bin/activate
python -m pytest api/tests/ -v
```

Expected: all tests pass. The mock batcher returns `([], 5.0)` so no detections are iterated — metric observation calls happen but counters stay 0. That's fine.

- [ ] **Step 5: Manual validation**

Start the API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

In a second terminal:
```bash
curl -s http://localhost:8080/metrics | grep -E "inference_duration|detection_total"
```

Expected: both metric names appear in the output (they'll be zero or just have type/help lines until a real /detect call happens).

- [ ] **Step 6: Commit**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git add api/main.py
git commit -m "feat: observe inference latency and detection count in detect endpoints"
```

---

### Task 3: Observe batcher queue depth and batch size

**Files:**
- Modify: `api/batcher.py`

**Context:** `api/batcher.py` has `submit()` (puts a frame on the queue) and `_drain_loop()` (drains up to `max_batch` items and sends to Triton). `batcher_queue_depth` is a Gauge — increment on enqueue, decrement after each future is resolved. `batcher_batch_size` is a Histogram — observe `len(items)` once per drain batch before sending to Triton.

- [ ] **Step 1: Import metrics in `api/batcher.py`**

Add to imports:
```python
from api.metrics import batcher_batch_size, batcher_queue_depth
```

- [ ] **Step 2: Increment queue depth in `submit()`**

Find the line `await self._queue.put((frame, future))` in `submit()`. Add the increment immediately after:
```python
        await self._queue.put((frame, future))
        batcher_queue_depth.inc()
```

- [ ] **Step 3: Observe batch size and decrement queue depth in `_drain_loop()`**

Find the section in `_drain_loop()` where `items` is built and Triton is called. It looks like:
```python
            frames = np.stack([item[0] for item in items])
            futures = [item[1] for item in items]

            try:
                dets_list, elapsed_ms = await loop.run_in_executor(
                    None, self._triton.infer_batch, frames
                )
```

Add `batcher_batch_size.observe(len(items))` before the `try` block:
```python
            frames = np.stack([item[0] for item in items])
            futures = [item[1] for item in items]
            batcher_batch_size.observe(len(items))

            try:
                dets_list, elapsed_ms = await loop.run_in_executor(
                    None, self._triton.infer_batch, frames
                )
```

Then find where futures are resolved (the `for future, dets in zip(futures, dets_list)` loop) and decrement after each resolution:
```python
                for future, dets in zip(futures, dets_list):
                    if not future.done():
                        future.set_result((dets, per_item_ms))
                        batcher_queue_depth.dec()
```

Also decrement on exception (in the `except BaseException` block), after setting exceptions on futures:
```python
            except BaseException as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                        batcher_queue_depth.dec()
                if isinstance(e, (asyncio.CancelledError, GeneratorExit)):
                    raise
```

- [ ] **Step 4: Run full test suite**

```bash
cd /home/sai_taran/nvidia-traffic-analytics && source venv/bin/activate
python -m pytest api/tests/ -v
```

Expected: all tests pass. The batcher tests use a mock triton; the metric calls fire but don't affect test assertions.

- [ ] **Step 5: Manual validation**

Start the API and check metrics:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
curl -s http://localhost:8080/metrics | grep -E "batcher_queue_depth|batcher_batch_size"
```

Expected: both metric names appear in output.

- [ ] **Step 6: Commit**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git add api/batcher.py
git commit -m "feat: observe batcher queue depth and batch size in AsyncBatcher"
```

---

### Task 4: Prometheus config + Grafana dashboard

**Files:**
- Modify: `serving/prometheus.yml`
- Create: `serving/grafana/dashboards/fastapi.json`

**Context:** Prometheus runs inside Docker Compose. FastAPI runs on the host at port 8080. `host.docker.internal` is Docker Desktop's magic hostname for reaching the host from inside a container (works on Windows/Mac with Docker Desktop). The existing Triton dashboard uses `"datasource": "Prometheus"` in targets with `"datasource": null` at panel level — match this pattern exactly.

- [ ] **Step 1: Update `serving/prometheus.yml`**

Replace the entire file with:
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

- [ ] **Step 2: Create `serving/grafana/dashboards/fastapi.json`**

```json
{
  "title": "FastAPI Gateway",
  "uid": "fastapi-gateway",
  "version": 1,
  "schemaVersion": 38,
  "refresh": "5s",
  "time": {"from": "now-5m", "to": "now"},
  "panels": [
    {
      "id": 1,
      "title": "Batcher Queue Depth",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "datasource": null,
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "batcher_queue_depth",
          "legendFormat": "queue depth",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {"unit": "short", "color": {"mode": "palette-classic"}}
      },
      "options": {"tooltip": {"mode": "single"}}
    },
    {
      "id": 2,
      "title": "Detections by Class (rate/min)",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "datasource": null,
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(detection_total[1m])",
          "legendFormat": "{{class_name}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {"defaults": {"unit": "short"}},
      "options": {"fillOpacity": 50, "tooltip": {"mode": "multi"}}
    },
    {
      "id": 3,
      "title": "Inference Latency p50 / p95",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "datasource": null,
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "histogram_quantile(0.5, rate(inference_duration_seconds_bucket[1m]))",
          "legendFormat": "p50",
          "refId": "A"
        },
        {
          "datasource": "Prometheus",
          "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[1m]))",
          "legendFormat": "p95",
          "refId": "B"
        }
      ],
      "fieldConfig": {"defaults": {"unit": "s"}},
      "options": {"tooltip": {"mode": "multi"}}
    },
    {
      "id": 4,
      "title": "Triton GPU Utilization",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
      "datasource": null,
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "nv_gpu_utilization",
          "legendFormat": "GPU %",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {"unit": "percent", "min": 0, "max": 100,
                     "color": {"mode": "fixed", "fixedColor": "green"}}
      },
      "options": {"tooltip": {"mode": "single"}}
    },
    {
      "id": 5,
      "title": "Requests per Second (/detect)",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
      "datasource": null,
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(http_requests_total{handler=\"/detect\"}[1m])",
          "legendFormat": "req/s",
          "refId": "A"
        }
      ],
      "fieldConfig": {"defaults": {"unit": "reqps"}},
      "options": {"tooltip": {"mode": "single"}}
    },
    {
      "id": 6,
      "title": "Batch Size Distribution",
      "type": "barchart",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
      "datasource": null,
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(batcher_batch_size_bucket[1m])",
          "legendFormat": "le={{le}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {"defaults": {"unit": "short"}},
      "options": {"tooltip": {"mode": "multi"}}
    }
  ]
}
```

- [ ] **Step 3: Verify Grafana provisioning picks up the new dashboard**

The existing `serving/grafana/provisioning/dashboards/dashboard.yml` should already point to the dashboards directory. Verify:
```bash
cat /home/sai_taran/nvidia-traffic-analytics/serving/grafana/provisioning/dashboards/dashboard.yml
```

It should contain something like `path: /etc/grafana/dashboards`. The new `fastapi.json` in that folder will be auto-provisioned when Grafana restarts.

- [ ] **Step 4: Commit**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git add serving/prometheus.yml serving/grafana/dashboards/fastapi.json
git commit -m "feat: add FastAPI scrape to Prometheus and 6-panel Grafana dashboard"
```

- [ ] **Step 5: Integration test (requires Docker Compose running)**

Start the stack:
```bash
cd /home/sai_taran/nvidia-traffic-analytics/serving
docker compose up -d
```

In a second terminal, start FastAPI:
```bash
cd /home/sai_taran/nvidia-traffic-analytics && source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Verify Prometheus sees FastAPI:
```bash
# Check target health at http://localhost:9090/targets — fastapi target should be UP
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -A3 "fastapi"
```

Open Grafana at `http://localhost:3000` → Dashboards → "FastAPI Gateway" — all 6 panels should load. Run the Locust test to populate metrics:
```bash
locust -f api/locust/locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8080
```

Watch the Grafana dashboard update live.

---

## Self-Review

**Spec coverage:**
- ✅ `api/metrics.py` — 4 metrics defined (Task 1)
- ✅ `/metrics` endpoint via instrumentator (Task 1)
- ✅ `inference_duration_seconds` observed in seconds — `elapsed_ms / 1000` (Task 2)
- ✅ `detection_total` by class_name incremented (Task 2)
- ✅ `batcher_queue_depth` inc on submit, dec on resolve (Task 3)
- ✅ `batcher_batch_size` observed per drain (Task 3)
- ✅ `serving/prometheus.yml` — FastAPI scrape job added (Task 4)
- ✅ 6-panel Grafana dashboard (Task 4): queue depth, detections by class, p50/p95 latency, GPU util, req/s, batch size distribution
- ✅ Datasource pattern matches existing `triton.json` (`"datasource": "Prometheus"` in targets)

**No placeholders found.**

**Type consistency:** `detection_total.labels(class_name=det.class_name).inc()` — `det.class_name` matches `Detection.class_name: str` defined in `api/schemas.py`. `batcher_queue_depth` imported identically in both `api/main.py` (Task 2) and `api/batcher.py` (Task 3) — no conflict since it's a module-level singleton from `api/metrics.py`.
