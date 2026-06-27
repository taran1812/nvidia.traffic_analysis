# Week 6: Polish and Publish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize FastAPI into Docker, wire a single `docker compose up` full stack (Triton + FastAPI + Prometheus + Grafana), and update the README for GitHub publish.

**Architecture:** FastAPI moves from host process to a Docker service built from a `Dockerfile` at project root. A new root `docker-compose.yml` brings up all four services. Prometheus reaches FastAPI via the Docker internal network (`fastapi:8000`) instead of `host.docker.internal`. `serving/docker-compose.yml` stays untouched for Triton-only use. `TRITON_URL` env var makes the Triton URL configurable so the same image works locally and in Docker.

**Tech Stack:** Docker, Docker Compose 3.8, Python 3.10-slim, FastAPI 0.138, uvicorn 0.49, tritonclient[http] 2.70

---

## File Map

| File | Action |
|---|---|
| `api/requirements.txt` | Create — pinned deps for Docker build |
| `Dockerfile` | Create — FastAPI container |
| `api/main.py` | Modify — read `TRITON_URL` env var instead of hardcoded `localhost:8000` |
| `api/tests/test_main.py` | Modify — add `test_triton_url_from_env` |
| `docker-compose.yml` | Create at project root — full stack |
| `serving/prometheus.yml` | Modify — change fastapi target from `host.docker.internal:8081` to `fastapi:8000` |
| `README.md` | Modify — mark weeks done, update setup section |

---

### Task 1: `api/requirements.txt`

**Files:**
- Create: `api/requirements.txt`

- [ ] **Step 1: Create `api/requirements.txt`**

```
fastapi==0.138.1
uvicorn==0.49.0
tritonclient[http]==2.70.0
prometheus-fastapi-instrumentator==8.0.2
prometheus_client==0.25.0
opencv-python-headless==4.13.0.92
httpx==0.28.1
numpy==2.2.6
pydantic==2.13.4
```

Note: `opencv-python-headless` (not `opencv-python`) — no GUI/display deps needed in container.

- [ ] **Step 2: Verify installable**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
source venv/bin/activate
pip install -r api/requirements.txt --dry-run 2>&1 | tail -3
```

Expected: `Would install ...` or `Requirement already satisfied` for all packages. No errors.

- [ ] **Step 3: Commit**

```bash
git add api/requirements.txt
git commit -m "feat: add api/requirements.txt for Docker build"
```

---

### Task 2: `TRITON_URL` env var in `api/main.py`

**Files:**
- Modify: `api/main.py` (line 19)
- Modify: `api/tests/test_main.py`

`TritonClient` is currently hardcoded to `localhost:8000`. Inside Docker it must reach `triton:8000` via the compose network. Read from env var with local fallback.

- [ ] **Step 1: Write failing test**

Add this test to `api/tests/test_main.py` (after existing imports, no new imports needed — `patch`, `AsyncMock`, `TestClient` already imported):

```python
def test_triton_url_from_env(monkeypatch):
    monkeypatch.setenv("TRITON_URL", "triton:8000")
    with patch('api.main.TritonClient') as mock_tc, \
         patch('api.main.AsyncBatcher') as mock_batcher_cls:
        mock_batcher_cls.return_value.start = AsyncMock()
        mock_batcher_cls.return_value.stop = AsyncMock()
        from api.main import app
        with TestClient(app):
            pass
    mock_tc.assert_called_once_with(url="triton:8000")
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
source venv/bin/activate
python -m pytest api/tests/test_main.py::test_triton_url_from_env -v
```

Expected: FAIL — `AssertionError: expected call with url='triton:8000' but got url='localhost:8000'`

- [ ] **Step 3: Update `api/main.py`**

Add `import os` at the top (after `from contextlib import asynccontextmanager`), then change line 19:

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.batcher import AsyncBatcher
from api.metrics import detection_total, inference_duration_seconds
from api.preprocess import load_from_bytes, load_from_url, preprocess
from api.schemas import DetectResponse, DetectURLRequest
from api.triton import TritonClient

_triton: TritonClient | None = None
_batcher: AsyncBatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _triton, _batcher
    _triton = TritonClient(url=os.getenv("TRITON_URL", "localhost:8000"))
    _batcher = AsyncBatcher(triton=_triton)
    await _batcher.start()
    yield
    await _batcher.stop()
```

Only change: line 19 `url="localhost:8000"` → `url=os.getenv("TRITON_URL", "localhost:8000")` and add `import os`.

- [ ] **Step 4: Run all tests**

```bash
python -m pytest api/tests/ -v 2>&1 | tail -10
```

Expected: all tests pass including `test_triton_url_from_env`.

- [ ] **Step 5: Commit**

```bash
git add api/main.py api/tests/test_main.py
git commit -m "feat: read TRITON_URL from env var, default localhost:8000"
```

---

### Task 3: `Dockerfile` + `.dockerignore`

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore` at project root**

```
venv/
.venv/
__pycache__/
*.pyc
*.egg-info/
.git/
serving/model_repository/
*.engine
*.onnx
*.pt
docs/
*.jpg
*.png
*.csv
```

This keeps the build context small — excludes venv (~1GB), TRT engine, ONNX/PT weights, and serving stack from being sent to the Docker daemon.

- [ ] **Step 2: Create `Dockerfile` at project root**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build image**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
docker build -t traffic-api .
```

Expected: build succeeds, final line `Successfully built <id>` or `naming to docker.io/library/traffic-api`.

- [ ] **Step 4: Smoke test — imports load**

```bash
docker run --rm traffic-api python -c "from api.main import app; print('ok')"
```

Expected output: `ok`

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile and .dockerignore for FastAPI service"
```

---

### Task 4: Root `docker-compose.yml`

**Files:**
- Create: `docker-compose.yml` (project root)

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.08-py3
    command: tritonserver --model-repository=/models --log-verbose=0
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./serving/model_repository:/models
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

  fastapi:
    build: .
    ports:
      - "8081:8000"
    environment:
      - TRITON_URL=triton:8000
    depends_on:
      triton:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./serving/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - triton
      - fastapi

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
      - GF_AUTH_DISABLE_LOGIN_FORM=true
    volumes:
      - ./serving/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./serving/grafana/dashboards:/etc/grafana/dashboards:ro
    depends_on:
      - prometheus
```

Note: FastAPI is exposed on host port `8081` (Triton occupies `8000` externally). Internally on the Docker network, FastAPI listens on `8000` — Prometheus reaches it as `fastapi:8000`.

- [ ] **Step 2: Validate compose file syntax**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
docker compose config > /dev/null && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add root docker-compose.yml for full stack (Triton + FastAPI + Prometheus + Grafana)"
```

---

### Task 5: Update `serving/prometheus.yml` — fix fastapi scrape target

**Files:**
- Modify: `serving/prometheus.yml`

Currently targets `host.docker.internal:8081`. In the new compose, FastAPI is reachable at `fastapi:8000` via Docker network.

- [ ] **Step 1: Update target**

Replace `serving/prometheus.yml` content:

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
      - targets: ['fastapi:8000']
```

- [ ] **Step 2: Commit**

```bash
git add serving/prometheus.yml
git commit -m "fix: prometheus fastapi target uses docker service name fastapi:8000"
```

---

### Task 6: Update `README.md`

**Files:**
- Modify: `README.md`

Three changes: (a) mark Week 5 items done, (b) update Week 6 checklist, (c) update setup section to show `docker compose up`.

- [ ] **Step 1: Mark Week 5 done**

Find the Week 5 section and update:

```markdown
### ✅ Week 5: Observability

- [x] Prometheus scraping: Triton metrics + custom FastAPI metrics
- [x] Grafana dashboard: req/s, inference latency p50/p95, detections by class, batcher queue depth, batch size distribution, GPU utilization
- [x] Custom metrics: `detections_total`, `inference_duration_seconds`, `batcher_queue_depth`, `batcher_batch_size`
```

(Remove the screenshot and guardrail items — screenshot is optional/manual, guardrail was out of scope.)

- [ ] **Step 2: Update Week 6 section**

```markdown
### ✅ Week 6: Polish and publish

- [x] FastAPI containerized — `Dockerfile` at project root
- [x] Root `docker-compose.yml` — one-command full stack
- [x] README updated with final benchmark results
- [x] Published to GitHub
```

- [ ] **Step 3: Update setup section**

Find the "Run full pipeline (Week 6+)" section and update:

```markdown
### Run full stack

```bash
git clone https://github.com/taran1812/nvidia-traffic-analytics
cd nvidia-traffic-analytics
docker compose up
```

FastAPI: http://localhost:8081
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 → Dashboards → FastAPI Gateway
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: mark weeks 5-6 complete, update setup for docker compose up"
```

---

### Task 7: Publish to GitHub

**Prerequisites:** GitHub repo `taran1812/nvidia-traffic-analytics` must exist (create at github.com if not already created — make it public).

- [ ] **Step 1: Add remote**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git remote add origin https://github.com/taran1812/nvidia-traffic-analytics.git
```

- [ ] **Step 2: Push**

```bash
git push -u origin master
```

Expected: all commits pushed, branch `master` tracks `origin/master`.

- [ ] **Step 3: Verify**

Open `https://github.com/taran1812/nvidia-traffic-analytics` in browser — confirm README renders and all files are present.
