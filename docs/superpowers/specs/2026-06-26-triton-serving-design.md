# Week 3: Triton Inference Server Design

**Date:** 2026-06-26  
**Scope:** Serve `yolov8n_trt10.engine` via Triton Inference Server, benchmark batch throughput, expose metrics to Prometheus + Grafana.

---

## Goal

Deploy Triton as a standalone inference server for the YOLOv8n TRT engine. Run a Python benchmark client that sweeps batch sizes 1/4/8/16 and measures latency + throughput. Stand up Prometheus + Grafana alongside Triton via Docker Compose and capture a live dashboard screenshot during the benchmark run.

---

## Architecture

```
WSL host
├── serving/benchmark.py         ← Python client (tritonclient.http)
│       │
│       └── HTTP :8000
│
└── docker-compose.yml
    ├── triton (:8000/:8001/:8002)   ← serves yolov8n TRT engine
    ├── prometheus (:9090)           ← scrapes triton:8002/metrics every 5s
    └── grafana (:3000)             ← pre-provisioned dashboard, anon access
```

Engine file (`yolov8n_trt10.engine`) mounted at runtime — not baked into image.

---

## Directory Structure

```
serving/
├── docker-compose.yml
├── prometheus.yml
├── benchmark.py
├── results/                        # gitignored — benchmark JSON output
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml
│   │   └── dashboards/dashboard.yml
│   └── dashboards/triton.json
└── model_repository/
    └── yolov8n/
        ├── config.pbtxt
        └── 1/
            └── model.plan          # symlink → yolov8n_trt10.engine (mounted)
```

---

## Triton Model Config (`config.pbtxt`)

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

---

## Docker Compose

**triton** — `nvcr.io/nvidia/tritonserver:24.08-py3`
- `--gpus all`
- `--model-repository=/models`
- ports: `8000` (HTTP), `8001` (gRPC), `8002` (metrics)
- volumes: `./model_repository:/models`, `../yolov8n_trt10.engine:/models/yolov8n/1/model.plan`

**prometheus** — `prom/prometheus:latest`
- scrapes `triton:8002/metrics` every 5s
- port `9090`

**grafana** — `grafana/grafana:latest`
- port `3000`
- anonymous access enabled (dev only)
- auto-provisioned datasource (Prometheus) + dashboard

---

## Benchmark Client (`benchmark.py`)

1. Load 200 frames from `pipeline/data/bus_test.mp4` or `pipeline/output/traffic_out.ogv` via OpenCV
2. Resize each frame to 640×640, normalize to FP32 [0,1], transpose to CHW
3. Warm up: 10 requests at batch_size=1
4. Sweep `batch_sizes = [1, 4, 8, 16]` — 100 iterations each
5. Measure per-batch: mean latency (ms), p50, p95, throughput (frames/sec)
6. Print results table to stdout
7. Write `serving/results/benchmark_YYYY-MM-DD.json`

Uses `tritonclient[http]` pip package. Runs on WSL host (not inside Docker).

---

## Prometheus Metrics (Triton built-in)

Key metrics scraped at `triton:8002/metrics`:

| Metric | What it shows |
|---|---|
| `nv_inference_request_success` | total successful requests |
| `nv_inference_queue_duration_us` | time requests wait in queue |
| `nv_inference_compute_infer_duration_us` | actual GPU compute time |
| `nv_gpu_utilization` | GPU util % |
| `nv_gpu_memory_used_bytes` | VRAM usage |

---

## Grafana Dashboard

Pre-provisioned JSON with 4 panels:
- GPU utilization % (gauge)
- Inference throughput req/sec (time series)
- Inference latency p50/p95 ms (time series)
- Queue depth (time series)

---

## Success Criteria

- [ ] `docker compose up` starts all 3 services cleanly
- [ ] Triton `/v2/health/ready` returns 200
- [ ] `benchmark.py` completes batch sweep, prints results table
- [ ] Grafana dashboard at `localhost:3000` shows live metrics during benchmark
- [ ] Screenshot of dashboard saved to `serving/results/grafana_screenshot.png`
- [ ] Benchmark JSON committed to repo

---

## Out of Scope

- Triton gRPC client (HTTP sufficient for benchmarking)
- Authentication on Grafana
- Triton model ensemble
- Integration with DeepStream pipeline (Week 4+)
