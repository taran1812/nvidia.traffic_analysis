# Triton Inference Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy YOLOv8n TRT engine on Triton Inference Server with Docker Compose (Triton + Prometheus + Grafana), benchmark batch sizes 1/4/8/16, and capture a live Grafana dashboard screenshot during the benchmark run.

**Architecture:** Python benchmark client on WSL host hits Triton HTTP API at `localhost:8000`. Docker Compose runs Triton (serving TRT engine), Prometheus (scraping Triton metrics every 5s), and Grafana (pre-provisioned dashboard). Engine file mounted at runtime — not baked into image.

**Tech Stack:** Triton Inference Server 24.08, Docker Compose, Prometheus, Grafana, `tritonclient[http]`, OpenCV, numpy, Python 3.10.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `serving/model_repository/yolov8n/config.pbtxt` | Create | Triton model config — TRT backend, dynamic batching, I/O shapes |
| `serving/model_repository/yolov8n/1/.gitkeep` | Create | Keeps directory in git; model.plan mounted at runtime |
| `serving/docker-compose.yml` | Create | Triton + Prometheus + Grafana services |
| `serving/prometheus.yml` | Create | Scrape config targeting triton:8002/metrics |
| `serving/grafana/provisioning/datasources/prometheus.yml` | Create | Auto-provision Prometheus datasource |
| `serving/grafana/provisioning/dashboards/dashboard.yml` | Create | Auto-provision dashboard from file |
| `serving/grafana/dashboards/triton.json` | Create | 4-panel Grafana dashboard JSON |
| `serving/benchmark.py` | Create | Batch sweep benchmark client |
| `serving/tests/test_benchmark.py` | Create | Unit tests for frame preprocessing |
| `.gitignore` | Modify | Add `serving/results/` |

---

## Task 1: Directory Scaffolding + .gitignore

**Files:**
- Create: `serving/model_repository/yolov8n/1/.gitkeep`
- Create: `serving/grafana/provisioning/datasources/.gitkeep`
- Create: `serving/grafana/provisioning/dashboards/.gitkeep`
- Create: `serving/grafana/dashboards/.gitkeep`
- Create: `serving/results/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p serving/model_repository/yolov8n/1
mkdir -p serving/grafana/provisioning/datasources
mkdir -p serving/grafana/provisioning/dashboards
mkdir -p serving/grafana/dashboards
mkdir -p serving/tests
mkdir -p serving/results
touch serving/model_repository/yolov8n/1/.gitkeep
touch serving/results/.gitkeep
touch serving/tests/__init__.py
```

- [ ] **Step 2: Add serving/results/ to .gitignore**

Add to `.gitignore`:
```
# serving outputs
serving/results/*.json
serving/results/*.png
```

- [ ] **Step 3: Commit**

```bash
git add serving/ .gitignore
git commit -m "feat: scaffold serving directory structure"
```

---

## Task 2: Triton Model Config

**Files:**
- Create: `serving/model_repository/yolov8n/config.pbtxt`

- [ ] **Step 1: Write config.pbtxt**

`serving/model_repository/yolov8n/config.pbtxt`:
```protobuf
name: "yolov8n"
backend: "tensorrt"
max_batch_size: 16

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }
]
```

> `dims` excludes the batch dimension — Triton adds it automatically up to `max_batch_size`.
> Input/output names must match the ONNX export from ultralytics (`images` → `output0`).

- [ ] **Step 2: Commit**

```bash
git add serving/model_repository/yolov8n/config.pbtxt
git commit -m "feat: add Triton model config for YOLOv8n TRT backend"
```

---

## Task 3: Docker Compose + Prometheus Config

**Files:**
- Create: `serving/docker-compose.yml`
- Create: `serving/prometheus.yml`

- [ ] **Step 1: Write docker-compose.yml**

`serving/docker-compose.yml`:
```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.08-py3
    command: tritonserver --model-repository=/models --log-verbose=0
    ports:
      - "8000:8000"   # HTTP
      - "8001:8001"   # gRPC
      - "8002:8002"   # metrics
    volumes:
      - ./model_repository:/models
      - ../yolov8n_trt10.engine:/models/yolov8n/1/model.plan:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 12

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - triton

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
      - GF_AUTH_DISABLE_LOGIN_FORM=true
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/etc/grafana/dashboards:ro
    depends_on:
      - prometheus
```

- [ ] **Step 2: Write prometheus.yml**

`serving/prometheus.yml`:
```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['triton:8002']
```

- [ ] **Step 3: Commit**

```bash
git add serving/docker-compose.yml serving/prometheus.yml
git commit -m "feat: add Docker Compose stack for Triton + Prometheus + Grafana"
```

---

## Task 4: Grafana Provisioning + Dashboard

**Files:**
- Create: `serving/grafana/provisioning/datasources/prometheus.yml`
- Create: `serving/grafana/provisioning/dashboards/dashboard.yml`
- Create: `serving/grafana/dashboards/triton.json`

- [ ] **Step 1: Write Grafana datasource provisioning**

`serving/grafana/provisioning/datasources/prometheus.yml`:
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    access: proxy
```

- [ ] **Step 2: Write Grafana dashboard provisioning**

`serving/grafana/provisioning/dashboards/dashboard.yml`:
```yaml
apiVersion: 1

providers:
  - name: triton
    folder: ''
    type: file
    disableDeletion: false
    options:
      path: /etc/grafana/dashboards
```

- [ ] **Step 3: Write Grafana dashboard JSON**

`serving/grafana/dashboards/triton.json`:
```json
{
  "title": "Triton Inference Server",
  "uid": "triton-yolov8",
  "version": 1,
  "schemaVersion": 38,
  "refresh": "5s",
  "time": { "from": "now-5m", "to": "now" },
  "panels": [
    {
      "id": 1,
      "title": "GPU Utilization %",
      "type": "gauge",
      "gridPos": { "h": 8, "w": 6, "x": 0, "y": 0 },
      "fieldConfig": {
        "defaults": {
          "min": 0,
          "max": 100,
          "unit": "percent",
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green", "value": null },
              { "color": "yellow", "value": 60 },
              { "color": "red", "value": 90 }
            ]
          }
        }
      },
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "nv_gpu_utilization",
          "legendFormat": "GPU {{gpu_uuid}}"
        }
      ]
    },
    {
      "id": 2,
      "title": "Inference Throughput (req/sec)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 9, "x": 6, "y": 0 },
      "fieldConfig": {
        "defaults": { "unit": "reqps" }
      },
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(nv_inference_request_success{model=\"yolov8n\"}[30s])",
          "legendFormat": "requests/sec"
        }
      ]
    },
    {
      "id": 3,
      "title": "Inference Compute Latency (ms)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 9, "x": 15, "y": 0 },
      "fieldConfig": {
        "defaults": { "unit": "ms" }
      },
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "rate(nv_inference_compute_infer_duration_us{model=\"yolov8n\"}[30s]) / rate(nv_inference_request_success{model=\"yolov8n\"}[30s]) / 1000",
          "legendFormat": "compute latency ms"
        },
        {
          "datasource": "Prometheus",
          "expr": "rate(nv_inference_queue_duration_us{model=\"yolov8n\"}[30s]) / rate(nv_inference_request_success{model=\"yolov8n\"}[30s]) / 1000",
          "legendFormat": "queue latency ms"
        }
      ]
    },
    {
      "id": 4,
      "title": "GPU Memory Used (MB)",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 6, "x": 0, "y": 8 },
      "fieldConfig": {
        "defaults": { "unit": "decmbytes" }
      },
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "nv_gpu_memory_used_bytes / 1024 / 1024",
          "legendFormat": "VRAM used MB"
        }
      ]
    }
  ]
}
```

- [ ] **Step 4: Commit**

```bash
git add serving/grafana/
git commit -m "feat: add Grafana provisioning and Triton dashboard"
```

---

## Task 5: Validate Triton Starts + Loads Model

No new files. Validates the stack before writing the benchmark client.

- [ ] **Step 1: Verify engine file exists**

```bash
ls -lh yolov8n_trt10.engine
```

Expected: file exists, size ~9.6MB. If missing, rebuild:
```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics && docker run --gpus all --rm \
  --entrypoint trtexec \
  -v \"\$(pwd)/yolov8n.onnx:/models/yolov8n.onnx\" \
  -v \"\$(pwd):/output\" \
  deepstream-traffic \
  --onnx=/models/yolov8n.onnx \
  --saveEngine=/output/yolov8n_trt10.engine \
  --fp16 2>&1 | tail -5"
```

- [ ] **Step 2: Start Triton only**

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics/serving && docker compose up triton -d 2>&1"
```

- [ ] **Step 3: Wait for Triton ready (up to 2 min for engine load)**

```bash
wsl -d Ubuntu-22.04 -- bash -c "for i in \$(seq 1 24); do curl -sf http://localhost:8000/v2/health/ready && echo 'READY' && break || echo \"Attempt \$i — waiting...\"; sleep 5; done"
```

Expected: `READY` printed within 2 minutes.

- [ ] **Step 4: Verify model loaded**

```bash
wsl -d Ubuntu-22.04 -- bash -c "curl -s http://localhost:8000/v2/models/yolov8n | python3 -m json.tool"
```

Expected JSON with `"name": "yolov8n"` and `"state": "READY"`.

- [ ] **Step 5: Check metrics endpoint**

```bash
wsl -d Ubuntu-22.04 -- bash -c "curl -s http://localhost:8002/metrics | grep nv_gpu_utilization | head -3"
```

Expected: lines like `nv_gpu_utilization{...} 0.0`

- [ ] **Step 6: Start full stack**

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics/serving && docker compose up -d 2>&1"
```

- [ ] **Step 7: Verify Grafana**

Open `http://localhost:3000` in browser. Expected: Grafana loads without login, "Triton Inference Server" dashboard visible under Dashboards.

---

## Task 6: Benchmark Client (TDD)

**Files:**
- Create: `serving/tests/test_benchmark.py`
- Create: `serving/benchmark.py`

- [ ] **Step 1: Install tritonclient on WSL host**

```bash
wsl -d Ubuntu-22.04 -- bash -c "pip3 install tritonclient[http] 2>&1 | tail -3"
```

Expected: `Successfully installed tritonclient-...`

- [ ] **Step 2: Write failing tests**

`serving/tests/test_benchmark.py`:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import pytest
from benchmark import preprocess_frame, preprocess_image_file, compute_stats


def test_preprocess_frame_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.shape == (3, 640, 640)


def test_preprocess_frame_dtype():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.dtype == np.float32


def test_preprocess_frame_normalized():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_preprocess_image_file_output_shape():
    # bus.jpg is in the project root (one level up from serving/)
    img_path = os.path.join(os.path.dirname(__file__), '../../bus.jpg')
    if not os.path.exists(img_path):
        pytest.skip("bus.jpg not found")
    frames = preprocess_image_file(img_path, n=10)
    assert frames.shape == (10, 3, 640, 640)
    assert frames.dtype == np.float32


def test_compute_stats_percentiles():
    latencies = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_stats(latencies, batch_size=4)
    assert stats['batch_size'] == 4
    assert stats['mean_latency_ms'] == pytest.approx(30.0, abs=0.1)
    assert stats['p50_latency_ms'] == pytest.approx(30.0, abs=0.1)
    assert stats['p95_latency_ms'] == pytest.approx(48.0, abs=1.0)
    assert stats['throughput_fps'] == pytest.approx(4 * 1000 / 30.0, abs=0.1)


def test_compute_stats_throughput_scales_with_batch():
    latencies = np.array([10.0] * 100)
    stats1 = compute_stats(latencies, batch_size=1)
    stats4 = compute_stats(latencies, batch_size=4)
    assert stats4['throughput_fps'] == pytest.approx(4 * stats1['throughput_fps'], abs=0.1)
```

- [ ] **Step 3: Run tests — expect failures**

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics && python3 -m pytest serving/tests/test_benchmark.py -v 2>&1"
```

Expected: `ModuleNotFoundError: No module named 'benchmark'` or `ImportError`.

- [ ] **Step 4: Implement benchmark.py**

`serving/benchmark.py`:
```python
"""
Triton Inference Server benchmark for YOLOv8n.
Sweeps batch sizes [1, 4, 8, 16], measures latency + throughput.

Usage:
    python3 serving/benchmark.py
    python3 serving/benchmark.py --url localhost:8000 --video path/to/video.mp4
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tritonclient.http as httpclient

MODEL_NAME = "yolov8n"
MODEL_VERSION = "1"
BATCH_SIZES = [1, 4, 8, 16]
WARMUP_ITERS = 10
BENCH_ITERS = 100


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize BGR frame to 640x640, normalize to [0,1] FP32, transpose to CHW."""
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return frame.transpose(2, 0, 1)  # HWC → CHW


def preprocess_image_file(image_path: str, n: int = 200) -> np.ndarray:
    """Load one image, repeat n times. Returns (n, 3, 640, 640) FP32."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    frame = preprocess_frame(img)
    return np.stack([frame] * n)


def load_frames_from_video(video_path: str, n: int = 200) -> np.ndarray:
    """Load n frames from video. Loops if video shorter than n. Returns (n, 3, 640, 640) FP32."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(preprocess_frame(frame))
    cap.release()
    return np.stack(frames[:n])


def compute_stats(latencies: np.ndarray, batch_size: int) -> dict:
    """Compute latency stats and throughput from array of per-batch latencies (ms)."""
    return {
        "batch_size": batch_size,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": float(batch_size * 1000 / np.mean(latencies)),
    }


def infer_batch(client: httpclient.InferenceServerClient, frames: np.ndarray) -> float:
    """Send one batch to Triton, return latency in ms."""
    inputs = [httpclient.InferInput("images", list(frames.shape), "FP32")]
    inputs[0].set_data_from_numpy(frames)
    outputs = [httpclient.InferRequestedOutput("output0")]
    t0 = time.perf_counter()
    client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=inputs,
        outputs=outputs,
    )
    return (time.perf_counter() - t0) * 1000


def run_batch_benchmark(
    client: httpclient.InferenceServerClient,
    frames: np.ndarray,
    batch_size: int,
) -> dict:
    """Warm up then benchmark one batch size. Returns stats dict."""
    for _ in range(WARMUP_ITERS):
        infer_batch(client, frames[:batch_size])

    latencies = []
    n = len(frames)
    for i in range(BENCH_ITERS):
        start = (i * batch_size) % max(1, n - batch_size)
        batch = frames[start : start + batch_size]
        latencies.append(infer_batch(client, batch))

    return compute_stats(np.array(latencies), batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000", help="Triton HTTP URL")
    ap.add_argument("--video", default=None, help="Video file for frames (optional)")
    args = ap.parse_args()

    # resolve frame source — video > bus.jpg fallback
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.video and Path(args.video).exists():
        print(f"Loading frames from video: {args.video}")
        frames = load_frames_from_video(args.video)
    else:
        image_path = str(project_root / "bus.jpg")
        print(f"Loading frames from image: {image_path}")
        frames = preprocess_image_file(image_path)

    print(f"Frames loaded: {frames.shape}  dtype={frames.dtype}\n")

    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton not ready at {args.url}")
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' not ready")
    print(f"Triton ready. Model '{MODEL_NAME}' loaded.\n")

    results = []
    for batch_size in BATCH_SIZES:
        print(f"Benchmarking batch_size={batch_size}...", end=" ", flush=True)
        stats = run_batch_benchmark(client, frames, batch_size)
        results.append(stats)
        print(
            f"mean={stats['mean_latency_ms']:.1f}ms  "
            f"p50={stats['p50_latency_ms']:.1f}ms  "
            f"p95={stats['p95_latency_ms']:.1f}ms  "
            f"fps={stats['throughput_fps']:.1f}"
        )

    print("\n--- Benchmark Results ---")
    print(f"{'Batch':>6} {'Mean(ms)':>10} {'p50(ms)':>8} {'p95(ms)':>8} {'FPS':>8}")
    for r in results:
        print(
            f"{r['batch_size']:>6} "
            f"{r['mean_latency_ms']:>10.1f} "
            f"{r['p50_latency_ms']:>8.1f} "
            f"{r['p95_latency_ms']:>8.1f} "
            f"{r['throughput_fps']:>8.1f}"
        )

    out_dir = script_dir / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"benchmark_{datetime.now().strftime('%Y-%m-%d')}.json"
    payload = {
        "date": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "triton_url": args.url,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics && python3 -m pytest serving/tests/test_benchmark.py -v 2>&1"
```

Expected:
```
PASSED serving/tests/test_benchmark.py::test_preprocess_frame_shape
PASSED serving/tests/test_benchmark.py::test_preprocess_frame_dtype
PASSED serving/tests/test_benchmark.py::test_preprocess_frame_normalized
PASSED serving/tests/test_benchmark.py::test_preprocess_image_file_output_shape
PASSED serving/tests/test_benchmark.py::test_compute_stats_percentiles
PASSED serving/tests/test_benchmark.py::test_compute_stats_throughput_scales_with_batch
6 passed
```

- [ ] **Step 6: Commit**

```bash
git add serving/benchmark.py serving/tests/test_benchmark.py serving/tests/__init__.py
git commit -m "feat: add Triton benchmark client with batch sweep (TDD)"
```

---

## Task 7: Full Integration Run + Commit Results

- [ ] **Step 1: Start full stack (if not already running)**

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics/serving && docker compose up -d && echo 'Stack started'"
```

- [ ] **Step 2: Wait for Triton ready**

```bash
wsl -d Ubuntu-22.04 -- bash -c "for i in \$(seq 1 24); do curl -sf http://localhost:8000/v2/health/ready && echo ' — READY' && break || echo \"Attempt \$i — waiting 5s...\"; sleep 5; done"
```

Expected: `READY` within 2 minutes.

- [ ] **Step 3: Wait for Prometheus + Grafana**

```bash
wsl -d Ubuntu-22.04 -- bash -c "curl -sf http://localhost:9090/-/ready && echo 'Prometheus OK'; curl -sf http://localhost:3000/api/health && echo 'Grafana OK'"
```

Expected: both return OK.

- [ ] **Step 4: Run benchmark**

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /home/sai_taran/nvidia-traffic-analytics && python3 serving/benchmark.py 2>&1"
```

Expected output (values will vary by GPU):
```
Frames loaded: (200, 3, 640, 640)  dtype=float32

Triton ready. Model 'yolov8n' loaded.

Benchmarking batch_size=1... mean=X.Xms  p50=X.Xms  p95=X.Xms  fps=XXX.X
Benchmarking batch_size=4... mean=X.Xms  p50=X.Xms  p95=X.Xms  fps=XXX.X
Benchmarking batch_size=8... mean=X.Xms  p50=X.Xms  p95=X.Xms  fps=XXX.X
Benchmarking batch_size=16... mean=X.Xms  p50=X.Xms  p95=X.Xms  fps=XXX.X

--- Benchmark Results ---
 Batch   Mean(ms)  p50(ms)  p95(ms)      FPS
     1        ...      ...      ...      ...
     4        ...      ...      ...      ...
     8        ...      ...      ...      ...
    16        ...      ...      ...      ...

Results saved to serving/results/benchmark_2026-06-26.json
```

- [ ] **Step 5: Open Grafana and screenshot dashboard**

Open browser: `http://localhost:3000`

Navigate to Dashboards → "Triton Inference Server". Re-run benchmark to generate live traffic:
```bash
wsl -d Ubuntu-22.04 -- bash -c "for i in 1 2 3; do cd /home/sai_taran/nvidia-traffic-analytics && python3 serving/benchmark.py; done"
```

Take a screenshot of the Grafana dashboard showing live metrics. Save to:
`serving/results/grafana_screenshot.png`

- [ ] **Step 6: Update README — mark Week 3 done**

In `README.md`, update Week 3 checkboxes:
```markdown
### ✅ Week 3: Triton Inference Server

- [x] Deploy Triton 24.08 serving YOLOv8n TRT engine via Docker Compose
- [x] Configure dynamic batching (preferred batch [4, 8], max 16)
- [x] Benchmark batch sizes 1/4/8/16 — latency p50/p95 + throughput FPS
- [x] Prometheus scraping Triton /metrics (GPU util, latency, throughput)
- [x] Grafana dashboard with live metrics during benchmark run
- [x] Results committed to serving/results/
```

Also update the `Nvidia tools used` table to reflect actual TRT version:
```markdown
| TensorRT 10.3 | FP16 model optimization — engine rebuilt inside container for TRT 10.3 compatibility |
```

- [ ] **Step 7: Commit results and README**

```bash
git add serving/results/benchmark_2026-06-26.json README.md
git commit -m "feat: complete Week 3 Triton serving — benchmark results and Grafana dashboard"
```

---

## Known Pitfalls

| Issue | Fix |
|---|---|
| `model.plan` not found by Triton | Verify `yolov8n_trt10.engine` exists in project root; path in compose is `../yolov8n_trt10.engine` relative to `serving/` |
| TRT engine version mismatch | `tritonserver:24.08-py3` uses same TRT 10.3 as DeepStream 7.1 container — engine is compatible |
| Input shape mismatch | YOLOv8n from ultralytics: input `images` (3,640,640), output `output0` (84,8400). If wrong, inspect with `trtexec --onnx=yolov8n.onnx --printLayerInfo` |
| Grafana shows "No data" | Wait 30s after starting benchmark for Prometheus to scrape. Check `http://localhost:9090/targets` — triton target must be UP |
| `tritonclient` not found on host | `pip3 install tritonclient[http]` in WSL |
| docker compose GPU not available | Ensure NVIDIA Container Toolkit installed in WSL: `nvidia-container-toolkit` |
