# FastAPI Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI HTTP gateway in front of Triton — accepts image uploads or URLs, batches requests via asyncio background worker, returns detection JSON with metadata. Load tested with Locust.

**Architecture:** FastAPI app with lifespan-managed `AsyncBatcher`. Each request preprocesses image → submits to `asyncio.Queue`. Background task drains queue every 50ms (or at batch=16), calls Triton once per batch, resolves per-request Futures. Sync Triton call runs in thread pool executor to avoid blocking event loop.

**Tech Stack:** FastAPI, uvicorn, tritonclient[http], httpx, opencv-python-headless, numpy, locust, pytest, pytest-asyncio.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `api/__init__.py` | Create | Package marker |
| `api/requirements.txt` | Create | API dependencies |
| `api/labels.py` | Create | 80-class COCO label list |
| `api/schemas.py` | Create | Pydantic models: Detection, DetectResponse, DetectURLRequest |
| `api/preprocess.py` | Create | load_from_bytes, load_from_url, preprocess |
| `api/triton.py` | Create | TritonClient, parse_output (NMS post-processing) |
| `api/batcher.py` | Create | AsyncBatcher — queue + background drain task |
| `api/main.py` | Create | FastAPI app, lifespan, /detect, /detect/url, /health |
| `api/tests/__init__.py` | Create | Package marker |
| `api/tests/test_preprocess.py` | Create | Unit tests for preprocess functions |
| `api/tests/test_triton.py` | Create | Unit tests for parse_output |
| `api/tests/test_batcher.py` | Create | Unit tests for AsyncBatcher |
| `api/locust/__init__.py` | Create | Package marker |
| `api/locust/locustfile.py` | Create | Locust load test: 50 users, 60s |
| `api/locust/results/.gitkeep` | Create | Keep results dir in git |
| `pytest.ini` | Create | asyncio_mode = auto |

---

## Task 1: Scaffold + Static Files

**Files:**
- Create: `api/__init__.py`
- Create: `api/requirements.txt`
- Create: `api/labels.py`
- Create: `api/schemas.py`
- Create: `api/tests/__init__.py`
- Create: `api/locust/__init__.py`
- Create: `api/locust/results/.gitkeep`
- Create: `pytest.ini`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p api/tests api/locust/results
touch api/__init__.py api/tests/__init__.py api/locust/__init__.py api/locust/results/.gitkeep
```

- [ ] **Step 2: Create `api/requirements.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
httpx>=0.27.0
tritonclient[http]>=2.49.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
locust>=2.28.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Install dependencies**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
source venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart httpx "tritonclient[http]" opencv-python-headless locust pytest-asyncio -q
```

Expected: installs without error.

- [ ] **Step 4: Create `api/labels.py`**

```python
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
```

- [ ] **Step 5: Create `api/schemas.py`**

```python
from pydantic import BaseModel


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1


class DetectResponse(BaseModel):
    detections: list[Detection]
    inference_time_ms: float
    model: str
    image_size: list[int]  # [H, W]


class DetectURLRequest(BaseModel):
    url: str
```

- [ ] **Step 6: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 7: Commit**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git add api/ pytest.ini
git commit -m "feat: scaffold FastAPI gateway structure"
```

---

## Task 2: Preprocess (TDD)

**Files:**
- Create: `api/tests/test_preprocess.py`
- Create: `api/preprocess.py`

- [ ] **Step 1: Write failing tests in `api/tests/test_preprocess.py`**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock
from api.preprocess import load_from_bytes, preprocess


def test_preprocess_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess(frame)
    assert result.shape == (3, 640, 640)


def test_preprocess_dtype():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = preprocess(frame)
    assert result.dtype == np.float32


def test_preprocess_normalized():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    result = preprocess(frame)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_preprocess_channel_first():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = preprocess(frame)
    assert result.shape[0] == 3


def test_load_from_bytes_valid():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    result = load_from_bytes(bytes(buf))
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_load_from_bytes_invalid_raises():
    with pytest.raises(ValueError, match="Invalid image"):
        load_from_bytes(b"not an image")
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd /home/sai_taran/nvidia-traffic-analytics && source venv/bin/activate
python -m pytest api/tests/test_preprocess.py -v
```

Expected: `ImportError: cannot import name 'load_from_bytes' from 'api.preprocess'` (or ModuleNotFoundError).

- [ ] **Step 3: Create `api/preprocess.py`**

```python
import cv2
import httpx
import numpy as np


def load_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image: could not decode bytes")
    return img


def load_from_url(url: str) -> np.ndarray:
    response = httpx.get(url, timeout=10.0, follow_redirects=True)
    response.raise_for_status()
    return load_from_bytes(response.content)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to 640x640, BGR→RGB, normalize [0,1], return CHW float32."""
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
python -m pytest api/tests/test_preprocess.py -v
```

Expected:
```
PASSED api/tests/test_preprocess.py::test_preprocess_shape
PASSED api/tests/test_preprocess.py::test_preprocess_dtype
PASSED api/tests/test_preprocess.py::test_preprocess_normalized
PASSED api/tests/test_preprocess.py::test_preprocess_channel_first
PASSED api/tests/test_preprocess.py::test_load_from_bytes_valid
PASSED api/tests/test_preprocess.py::test_load_from_bytes_invalid_raises
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add api/preprocess.py api/tests/test_preprocess.py
git commit -m "feat: add image preprocess with TDD (load_from_bytes, preprocess)"
```

---

## Task 3: Triton Wrapper + NMS (TDD)

**Files:**
- Create: `api/tests/test_triton.py`
- Create: `api/triton.py`

- [ ] **Step 1: Write failing tests in `api/tests/test_triton.py`**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from api.triton import parse_output
from api.schemas import Detection


def test_parse_output_empty_returns_empty_list():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    result = parse_output(raw, conf_thresh=0.25, iou_thresh=0.45)
    assert result == [[]]


def test_parse_output_batch_size_matches():
    raw = np.zeros((3, 84, 8400), dtype=np.float32)
    result = parse_output(raw, conf_thresh=0.25, iou_thresh=0.45)
    assert len(result) == 3


def test_parse_output_detects_high_confidence_box():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    # anchor 0: cx=320, cy=320, w=100, h=100, class 2 (car) score=0.9
    raw[0, 0, 0] = 320.0
    raw[0, 1, 0] = 320.0
    raw[0, 2, 0] = 100.0
    raw[0, 3, 0] = 100.0
    raw[0, 4 + 2, 0] = 0.9
    result = parse_output(raw, conf_thresh=0.25, iou_thresh=0.45)
    assert len(result[0]) == 1
    assert result[0][0].class_name == "car"
    assert result[0][0].confidence == pytest.approx(0.9, abs=0.01)


def test_parse_output_bbox_normalized():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    raw[0, 0, 0] = 320.0
    raw[0, 1, 0] = 320.0
    raw[0, 2, 0] = 640.0
    raw[0, 3, 0] = 640.0
    raw[0, 4, 0] = 0.95
    result = parse_output(raw, conf_thresh=0.25, iou_thresh=0.45)
    assert len(result[0]) == 1
    x1, y1, x2, y2 = result[0][0].bbox
    assert 0.0 <= x1 <= 1.0
    assert 0.0 <= y1 <= 1.0
    assert 0.0 <= x2 <= 1.0
    assert 0.0 <= y2 <= 1.0


def test_parse_output_filters_low_confidence():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    raw[0, 0, 0] = 320.0
    raw[0, 1, 0] = 320.0
    raw[0, 2, 0] = 100.0
    raw[0, 3, 0] = 100.0
    raw[0, 4, 0] = 0.1  # below threshold
    result = parse_output(raw, conf_thresh=0.25, iou_thresh=0.45)
    assert result == [[]]
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
python -m pytest api/tests/test_triton.py -v
```

Expected: `ImportError: cannot import name 'parse_output' from 'api.triton'`.

- [ ] **Step 3: Create `api/triton.py`**

```python
import time
import numpy as np
import cv2
import tritonclient.http as httpclient
from api.labels import COCO_CLASSES
from api.schemas import Detection

MODEL_NAME = "yolov8n"
MODEL_VERSION = "1"


def parse_output(
    raw: np.ndarray,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> list[list[Detection]]:
    """Parse Triton output [N, 84, 8400] → list of Detection lists per image."""
    batch_size = raw.shape[0]
    results = []
    for n in range(batch_size):
        boxes_cxcywh = raw[n, :4, :].T   # [8400, 4]
        class_scores = raw[n, 4:, :].T    # [8400, 80]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        mask = confidences > conf_thresh
        if not mask.any():
            results.append([])
            continue

        boxes_f = boxes_cxcywh[mask]
        confs_f = confidences[mask].tolist()
        ids_f = class_ids[mask]

        # cx,cy,w,h → x,y,w,h for cv2 NMS
        boxes_xywh = boxes_f.copy()
        boxes_xywh[:, 0] = boxes_f[:, 0] - boxes_f[:, 2] / 2
        boxes_xywh[:, 1] = boxes_f[:, 1] - boxes_f[:, 3] / 2

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), confs_f, conf_thresh, iou_thresh
        )

        image_dets = []
        for idx in indices:
            i = int(idx)
            cx, cy, w, h = boxes_f[i]
            x1 = float(max(0.0, (cx - w / 2) / 640))
            y1 = float(max(0.0, (cy - h / 2) / 640))
            x2 = float(min(1.0, (cx + w / 2) / 640))
            y2 = float(min(1.0, (cy + h / 2) / 640))
            image_dets.append(Detection(
                class_name=COCO_CLASSES[int(ids_f[i])],
                confidence=float(confs_f[i]),
                bbox=[x1, y1, x2, y2],
            ))
        results.append(image_dets)
    return results


class TritonClient:
    def __init__(self, url: str = "localhost:8000"):
        self._client = httpclient.InferenceServerClient(url=url)

    def is_ready(self) -> bool:
        try:
            return self._client.is_server_ready() and self._client.is_model_ready(MODEL_NAME)
        except Exception:
            return False

    def infer_batch(self, frames: np.ndarray) -> list[list[Detection]]:
        """Send [N, 3, 640, 640] FP32 batch to Triton, return detections per image."""
        inp = httpclient.InferInput("images", list(frames.shape), "FP32")
        inp.set_data_from_numpy(frames)
        out = httpclient.InferRequestedOutput("output0")
        t0 = time.perf_counter()
        result = self._client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=[inp],
            outputs=[out],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        raw = result.as_numpy("output0")
        dets = parse_output(raw)
        return dets, elapsed_ms
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
python -m pytest api/tests/test_triton.py -v
```

Expected:
```
PASSED api/tests/test_triton.py::test_parse_output_empty_returns_empty_list
PASSED api/tests/test_triton.py::test_parse_output_batch_size_matches
PASSED api/tests/test_triton.py::test_parse_output_detects_high_confidence_box
PASSED api/tests/test_triton.py::test_parse_output_bbox_normalized
PASSED api/tests/test_triton.py::test_parse_output_filters_low_confidence
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add api/triton.py api/tests/test_triton.py
git commit -m "feat: add Triton wrapper with NMS post-processing (TDD)"
```

---

## Task 4: AsyncBatcher (TDD)

**Files:**
- Create: `api/tests/test_batcher.py`
- Create: `api/batcher.py`

- [ ] **Step 1: Write failing tests in `api/tests/test_batcher.py`**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock
from api.batcher import AsyncBatcher


def make_frame():
    return np.zeros((3, 640, 640), dtype=np.float32)


@pytest.mark.asyncio
async def test_single_submit_resolves():
    mock_triton = MagicMock()
    mock_triton.infer_batch.return_value = ([[]], 5.0)
    batcher = AsyncBatcher(triton=mock_triton, max_batch=16, drain_interval=0.01)
    await batcher.start()
    dets, ms = await batcher.submit(make_frame())
    assert dets == []
    assert isinstance(ms, float)
    await batcher.stop()


@pytest.mark.asyncio
async def test_multiple_submits_all_resolve():
    mock_triton = MagicMock()
    mock_triton.infer_batch.return_value = ([[], [], []], 10.0)
    batcher = AsyncBatcher(triton=mock_triton, max_batch=16, drain_interval=0.01)
    await batcher.start()
    tasks = [asyncio.create_task(batcher.submit(make_frame())) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    for dets, ms in results:
        assert isinstance(dets, list)
    await batcher.stop()


@pytest.mark.asyncio
async def test_queue_full_raises():
    mock_triton = MagicMock()
    mock_triton.infer_batch.return_value = ([[]], 5.0)
    batcher = AsyncBatcher(triton=mock_triton, max_batch=16, drain_interval=999, max_queue=2)
    await batcher.start()
    await batcher._queue.put((make_frame(), asyncio.get_event_loop().create_future()))
    await batcher._queue.put((make_frame(), asyncio.get_event_loop().create_future()))
    with pytest.raises(RuntimeError, match="queue full"):
        await batcher.submit(make_frame())
    await batcher.stop()
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
python -m pytest api/tests/test_batcher.py -v
```

Expected: `ImportError: cannot import name 'AsyncBatcher' from 'api.batcher'`.

- [ ] **Step 3: Create `api/batcher.py`**

```python
import asyncio
import numpy as np


class AsyncBatcher:
    def __init__(self, triton, max_batch: int = 16, drain_interval: float = 0.05, max_queue: int = 500):
        self._triton = triton
        self._max_batch = max_batch
        self._drain_interval = drain_interval
        self._max_queue = max_queue
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._drain_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, frame: np.ndarray) -> tuple[list, float]:
        if self._queue.qsize() >= self._max_queue:
            raise RuntimeError("queue full")
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((frame, future))
        return await future

    async def _drain_loop(self):
        loop = asyncio.get_event_loop()
        while self._running:
            await asyncio.sleep(self._drain_interval)
            if self._queue.empty():
                continue

            items = []
            while not self._queue.empty() and len(items) < self._max_batch:
                items.append(self._queue.get_nowait())

            if not items:
                continue

            frames = np.stack([item[0] for item in items])
            futures = [item[1] for item in items]

            try:
                dets_list, elapsed_ms = await loop.run_in_executor(
                    None, self._triton.infer_batch, frames
                )
                per_item_ms = elapsed_ms / len(items)
                for future, dets in zip(futures, dets_list):
                    if not future.done():
                        future.set_result((dets, per_item_ms))
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
python -m pytest api/tests/test_batcher.py -v
```

Expected:
```
PASSED api/tests/test_batcher.py::test_single_submit_resolves
PASSED api/tests/test_batcher.py::test_multiple_submits_all_resolve
PASSED api/tests/test_batcher.py::test_queue_full_raises
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add api/batcher.py api/tests/test_batcher.py
git commit -m "feat: add AsyncBatcher with queue + background drain task (TDD)"
```

---

## Task 5: FastAPI App

**Files:**
- Create: `api/main.py`

- [ ] **Step 1: Create `api/main.py`**

```python
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.batcher import AsyncBatcher
from api.preprocess import load_from_bytes, load_from_url, preprocess
from api.schemas import DetectResponse, DetectURLRequest
from api.triton import TritonClient

TRITON_URL = "localhost:8000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    triton = TritonClient(url=TRITON_URL)
    batcher = AsyncBatcher(triton=triton)
    await batcher.start()
    app.state.triton = triton
    app.state.batcher = batcher
    yield
    await batcher.stop()


app = FastAPI(title="Traffic Analytics API", lifespan=lifespan)


async def _run_detection(img_bgr, batcher: AsyncBatcher) -> DetectResponse:
    h, w = img_bgr.shape[:2]
    frame = preprocess(img_bgr)
    dets, elapsed_ms = await batcher.submit(frame)
    return DetectResponse(
        detections=dets,
        inference_time_ms=round(elapsed_ms, 2),
        model="yolov8n",
        image_size=[h, w],
    )


@app.post("/detect", response_model=DetectResponse)
async def detect_file(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = load_from_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        return await _run_detection(img, app.state.batcher)
    except RuntimeError as e:
        if "queue full" in str(e):
            raise HTTPException(status_code=429, detail="Server overloaded, retry later")
        raise HTTPException(status_code=503, detail="Inference service unavailable")


@app.post("/detect/url", response_model=DetectResponse)
async def detect_url(req: DetectURLRequest):
    try:
        img = load_from_url(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    try:
        return await _run_detection(img, app.state.batcher)
    except RuntimeError as e:
        if "queue full" in str(e):
            raise HTTPException(status_code=429, detail="Server overloaded, retry later")
        raise HTTPException(status_code=503, detail="Inference service unavailable")


@app.get("/health")
async def health():
    triton_status = "ready" if app.state.triton.is_ready() else "unavailable"
    return {"status": "ok", "triton": triton_status}
```

- [ ] **Step 2: Start server (Triton must be running)**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload &
sleep 3
```

- [ ] **Step 3: Smoke test health endpoint**

```bash
curl -s http://localhost:8080/health
```

Expected: `{"status":"ok","triton":"ready"}`

- [ ] **Step 4: Smoke test /detect with bus.jpg**

```bash
curl -s -X POST http://localhost:8080/detect \
  -F "file=@/home/sai_taran/nvidia-traffic-analytics/bus.jpg" | python3 -m json.tool
```

Expected: JSON with `detections` array (bus.jpg has a bus, people), `inference_time_ms`, `model: "yolov8n"`, `image_size`.

- [ ] **Step 5: Kill dev server**

```bash
kill %1 2>/dev/null || pkill -f "uvicorn api.main"
```

- [ ] **Step 6: Run all unit tests**

```bash
python -m pytest api/tests/ -v
```

Expected: 14 tests pass (6 preprocess + 5 triton + 3 batcher).

- [ ] **Step 7: Commit**

```bash
git add api/main.py
git commit -m "feat: add FastAPI gateway with /detect, /detect/url, /health endpoints"
```

---

## Task 6: Locust Load Test

**Files:**
- Create: `api/locust/locustfile.py`

- [ ] **Step 1: Create `api/locust/locustfile.py`**

```python
import os
from locust import HttpUser, task, between

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "../../bus.jpg")


class DetectUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://localhost:8080"

    def on_start(self):
        with open(IMAGE_PATH, "rb") as f:
            self._image_bytes = f.read()

    @task
    def detect(self):
        self.client.post(
            "/detect",
            files={"file": ("bus.jpg", self._image_bytes, "image/jpeg")},
        )
```

- [ ] **Step 2: Start FastAPI server**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8080 &
sleep 3
curl -s http://localhost:8080/health
```

Expected: `{"status":"ok","triton":"ready"}`

- [ ] **Step 3: Run Locust headless**

```bash
mkdir -p api/locust/results
locust -f api/locust/locustfile.py \
  --headless \
  -u 50 -r 5 -t 60s \
  --csv api/locust/results/locust_$(date +%F) \
  2>&1 | tee api/locust/results/locust_$(date +%F)_log.txt
```

Expected: runs for 60s. Final lines show RPS, failure count, p50/p95 latency.

- [ ] **Step 4: Kill server**

```bash
pkill -f "uvicorn api.main"
```

- [ ] **Step 5: Commit results and locustfile**

```bash
cd /home/sai_taran/nvidia-traffic-analytics
git add api/locust/locustfile.py
git add -f api/locust/results/
git commit -m "feat: add Locust load test + results (50 users, 60s)"
```

- [ ] **Step 6: Update README — mark Week 4 done**

In `README.md`, replace the Week 4 section:

```markdown
### ✅ Week 4: FastAPI gateway

- [x] FastAPI service accepting image file upload (multipart) and image URL
- [x] Async batcher: asyncio.Queue + background drain task, batch size up to 16
- [x] Route inference through Triton with NMS post-processing
- [x] Return detections as JSON: class, confidence, bbox (normalized), inference_time_ms
- [x] Load tested with Locust: 50 concurrent users, 60s
```

- [ ] **Step 7: Commit README**

```bash
git add README.md
git commit -m "docs: mark Week 4 complete in README"
```

---

## Known Pitfalls

| Issue | Fix |
|---|---|
| `tritonclient` HTTP calls block event loop | Already handled — `run_in_executor` in `_drain_loop` |
| `asyncio.Queue` created before event loop starts | Queue created in `__init__`; fine in Python 3.10+ |
| `parse_output` returns wrong class for index 0 | `COCO_CLASSES[0]` = "person" — verify with bus.jpg which always has people |
| Locust `--csv` creates multiple files | `_stats.csv`, `_stats_history.csv`, `_failures.csv` — all committed |
| Triton not running when API starts | `/health` returns `"triton": "unavailable"` but server still starts — requests will 503 |
