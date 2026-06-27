
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

## Project status

### ✅ Done — Week 1: TensorRT optimization

- [x] WSL2 + Ubuntu 22.04 environment setup
- [x] GPU passthrough verified (`nvidia-smi` in WSL2)
- [x] PyTorch 2.11 with CUDA 12.8 confirmed on RTX 3050 Ti
- [x] YOLOv8n inference running on GPU (51ms, 6 detections on test image)
- [x] ONNX export — yolov8n.onnx (12.3MB)
- [x] TensorRT FP16 engine build — yolov8n.engine (8.8MB, 88s compile)
- [x] 3-way benchmark: PyTorch vs ONNX vs TensorRT — **1.8× improvement**

### ✅ Week 2: DeepStream pipeline

- [x] Pull Nvidia DeepStream Docker image (`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`)
- [x] Build DeepStream pipeline: video file → TensorRT YOLOv8 → bounding boxes
- [x] Download UA-DETRAC traffic dataset
- [x] Add vehicle counting logic: sliding 60s window per class
- [x] Resolve dGPU compatibility issues, validate pipeline end-to-end

### ✅ Week 3: Triton Inference Server

- [x] Rebuild YOLOv8n TRT engine with dynamic batch (min=1, opt=8, max=16) via TRT 10.3
- [x] Configure Triton model repository — TRT backend, explicit batch (max_batch_size=0)
- [x] Docker Compose stack: Triton 24.08 + Prometheus + Grafana
- [x] Prometheus scraping Triton `/metrics` at 5s interval
- [x] Grafana dashboard: GPU util, throughput, compute latency, VRAM
- [x] Batch sweep benchmark (1/4/8/16) — peak 63 FPS at batch=4 on RTX 3050 Ti

### ✅ Week 4: FastAPI gateway

- [x] FastAPI service: POST /detect (file upload), POST /detect/url, GET /health
- [x] Async batching via AsyncBatcher — 50ms drain interval, up to 16 frames/batch
- [x] Full NMS post-processing: YOLOv8n output → detections with class names + normalized bboxes
- [x] Locust load test configured: 50 users, 60s, POST bus.jpg to /detect
- [x] 21 unit tests passing across preprocess, triton wrapper, batcher, and API endpoints

### 🔲 Week 5: Observability

- [ ] Prometheus scraping: Triton metrics + custom FastAPI metrics
- [ ] Grafana dashboard: FPS, GPU utilization, inference latency p50/p95, detection count over time
- [ ] Screenshot dashboard for README
- [ ] Add hallucination-style guardrail: flag low-confidence detections below threshold

### 🔲 Week 6: Polish and publish

- [ ] Docker Compose full stack: DeepStream + Triton + FastAPI + Prometheus + Grafana
- [ ] Write benchmark report: methodology, hardware, results, limitations
- [ ] Clean README with architecture diagram and results screenshots
- [ ] Publish on GitHub with one-command setup
- [ ] Write LinkedIn post: "Real-time traffic analytics on Nvidia DeepStream + TensorRT — benchmark results"
- [ ] Tag @NvidiaAI on LinkedIn

---

## Setup

### Prerequisites

- Windows 11 with WSL2 (Ubuntu 22.04)
- Nvidia RTX GPU with driver 525+
- Docker Desktop with WSL2 backend enabled
- Python 3.10+

### Install

```bash
git clone https://github.com/taran1812/nvidia-traffic-analytics
cd nvidia-traffic-analytics
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics onnx onnxruntime-gpu tensorrt-cu12 nvidia-modelopt[onnx]
```

### Run benchmark

```bash
python3 benchmark.py
```

### Run full pipeline (Week 6+)

```bash
docker compose up
```

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

## Resume bullet

> Optimized YOLOv8 inference using TensorRT FP16 quantization on RTX 3050 Ti; achieved 1.8× FPS improvement over PyTorch baseline (60 → 110 FPS) with 44% latency reduction — 99.1% of model nodes converted to FP16. Served via Triton Inference Server with GPU observability via Prometheus and Grafana.

---

## Tech stack

Python · PyTorch · YOLOv8 · ONNX · TensorRT · Nvidia DeepStream · Triton Inference Server · FastAPI · Prometheus · Grafana · Docker · WSL2 · CUDA 12.8
