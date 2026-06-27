
# nvidia-traffic-analytics

Real-time traffic video analytics pipeline built on Nvidia's production AI stack. Detects and counts vehicles in video streams using a TensorRT-optimized YOLOv8 model served through Triton Inference Server, with full GPU observability.

Built to benchmark the full Nvidia inference optimization chain — PyTorch → ONNX → TensorRT FP16 — on real hardware.

---

## Benchmark results

**Hardware:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)  
**Model:** YOLOv8n · 640×640 · 50 runs

| Mode | Latency | FPS | Speedup |
|---|---|---|---|
| PyTorch FP32 (baseline) | ~16ms | ~60 | 1.0× |
| ONNX FP32 | ~14ms | ~70 | ~1.2× |
| TensorRT FP16 | ~9ms | ~110 | **1.8×** |

TensorRT FP16 delivers **1.8× faster inference** over PyTorch baseline with **44% latency reduction**.  
99.1% of model nodes converted to FP16. Engine size: 8.8MB.

**Triton serving — batch sweep (RTX 3050 Ti, TRT 10.3 FP16):**

| Batch | Mean latency | p50 | p95 | Throughput |
|---|---|---|---|---|
| 1 | 20.8ms | 20.4ms | 22.9ms | 48 FPS |
| 4 | 63.7ms | 59.6ms | 88.0ms | **63 FPS** |
| 8 | 151ms | 130ms | 197ms | 53 FPS |
| 16 | 270ms | 244ms | 396ms | 59 FPS |

Peak throughput at batch=4 (63 FPS). Dynamic batching configured via Triton with preferred batch sizes [4, 8].

**FastAPI gateway — load test (50 concurrent users, 60s, RTX 3050 Ti):**

| Metric | Value |
|---|---|
| Requests | 1,865 |
| Error rate | 0% |
| Throughput | 31 req/s |
| p50 latency | 1,200ms |
| p95 latency | 1,500ms |
| p99 latency | 1,700ms |

End-to-end latency includes FastAPI, AsyncBatcher (50ms drain, up to 16 frames/batch), Triton inference, and NMS post-processing.

---

## Architecture

```
Video input (MP4 / RTSP)
        ↓
  DeepStream SDK          ← Nvidia video analytics pipeline
        ↓
  TensorRT FP16 engine    ← YOLOv8n optimized for RTX 3050 Ti
        ↓
  Triton Inference Server ← Production model serving
        ↓
  FastAPI gateway         ← REST API: accepts video/image, returns JSON
        ↓
  Prometheus + Grafana    ← GPU utilization, FPS, latency p95
```

---

## Nvidia tools used

| Tool | Purpose |
|---|---|
| TensorRT 10.3 | FP16 model optimization — engine rebuilt inside Triton 24.08 container for TRT 10.3 compatibility |
| Triton Inference Server | Production inference serving with dynamic batching |
| DeepStream SDK | GPU-accelerated video analytics pipeline |
| ModelOpt AutoCast | Automatic FP16 quantization (231/233 nodes) |

---

## Setup

### Prerequisites

- Windows 11 with WSL2 (Ubuntu 22.04)
- Nvidia RTX GPU with driver 525+
- Docker Desktop with WSL2 backend enabled
- Python 3.10+

### Install

```bash
git clone https://github.com/taran1812/nvidia.traffic_analysis
cd nvidia.traffic_analysis
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics onnx onnxruntime-gpu tensorrt-cu12 nvidia-modelopt[onnx]
```

### Run benchmark

```bash
python3 benchmark.py
```

### Run full stack

```bash
git clone https://github.com/taran1812/nvidia.traffic_analysis
cd nvidia.traffic_analysis
docker compose up
```

- FastAPI: http://localhost:8081/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 → Dashboards → FastAPI Gateway

---

## File structure

```
nvidia-traffic-analytics/
├── yolov8n.pt          # PyTorch model
├── yolov8n.onnx        # ONNX export (12.3MB)
├── yolov8n.engine      # TensorRT FP16 engine (8.8MB)
├── bus.jpg             # Test image
├── benchmark.py        # 3-way benchmark script (coming Week 1 cleanup)
├── pipeline/           # DeepStream pipeline (Week 2)
├── serving/            # Triton model repository (Week 3)
├── api/                # FastAPI gateway (Week 4)
├── observability/      # Prometheus + Grafana config (Week 5)
└── docker-compose.yml  # Full stack (Week 6)
```

---

## Tech stack

Python · PyTorch · YOLOv8 · ONNX · TensorRT · Nvidia DeepStream · Triton Inference Server · FastAPI · Prometheus · Grafana · Docker · WSL2 · CUDA 12.8
