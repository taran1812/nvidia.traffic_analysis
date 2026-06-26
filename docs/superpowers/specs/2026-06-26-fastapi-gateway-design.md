# FastAPI Gateway Design — Week 4

**Goal:** HTTP gateway in front of Triton that accepts image file uploads or URLs, batches requests asynchronously via a background worker, and returns structured detection JSON. Load tested with Locust.

---

## Architecture

```
POST /detect  (multipart file)  ─┐
POST /detect/url  (JSON URL)    ─┤→ preprocess → AsyncBatcher.submit()
                                  │                    ↓
                                  │             asyncio.Queue
                                  │                    ↓
                                  │      background task (wakes every 50ms or batch=16)
                                  │      drains queue → stacks tensor → Triton HTTP
                                  │                    ↓
                                  └──────── response: detections + metadata
```

**Tech stack:** FastAPI, uvicorn, tritonclient[http], httpx (URL fetch), opencv-python, numpy, locust.

---

## File Map

| File | Responsibility |
|---|---|
| `api/main.py` | FastAPI app, lifespan startup/shutdown, `/detect` and `/detect/url` endpoints |
| `api/batcher.py` | `AsyncBatcher` — asyncio.Queue + background drain task |
| `api/triton.py` | Triton client wrapper — infer batch, parse raw output, apply NMS |
| `api/preprocess.py` | Load image from bytes or URL, resize 640×640, normalize [0,1], CHW |
| `api/schemas.py` | Pydantic models: `DetectionResult`, `DetectResponse`, `DetectURLRequest` |
| `api/labels.py` | 80-class COCO label list, index → class name |
| `api/tests/test_preprocess.py` | Unit tests for preprocess functions |
| `api/tests/test_batcher.py` | Unit tests for AsyncBatcher queue + batching logic |
| `api/locust/locustfile.py` | Locust load test: 50 users, 60s, POST bus.jpg to /detect |

---

## Component Details

### AsyncBatcher (`api/batcher.py`)

- `asyncio.Queue` holds `(frame: np.ndarray, future: asyncio.Future)` tuples
- `submit(frame)` → puts item on queue, returns awaitable Future
- Background task loop: wakes every 50ms OR when queue depth hits 16
- Drains up to 16 items, stacks into `[N, 3, 640, 640]` tensor
- Single Triton infer call → splits response back to individual Futures
- Batcher started/stopped via FastAPI lifespan context manager

### Triton Wrapper (`api/triton.py`)

- `InferenceServerClient` pointed at `localhost:8000`
- Input: `[N, 3, 640, 640]` FP32, name `images`
- Output: `[N, 84, 8400]` FP32, name `output0`
- Post-process: first 4 of 84 channels = cx,cy,w,h → convert to x1,y1,x2,y2; remaining 80 = class scores
- NMS: confidence threshold 0.25, IoU threshold 0.45 (YOLOv8 defaults)
- Returns `List[List[Detection]]` — outer list per image, inner per object

### Preprocess (`api/preprocess.py`)

- `load_from_bytes(data: bytes) -> np.ndarray` — cv2.imdecode
- `load_from_url(url: str) -> np.ndarray` — httpx.get, then imdecode
- `preprocess(img: np.ndarray) -> np.ndarray` — resize 640×640, BGR→RGB, /255, CHW, float32

### Schemas (`api/schemas.py`)

```python
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

### Endpoints (`api/main.py`)

```
POST /detect
  Content-Type: multipart/form-data
  Body: file=<image bytes>
  Response: DetectResponse

POST /detect/url
  Content-Type: application/json
  Body: {"url": "https://..."}
  Response: DetectResponse

GET /health
  Response: {"status": "ok", "triton": "ready"|"unavailable"}
```

---

## Load Test (`api/locust/locustfile.py`)

- Task: POST `bus.jpg` to `/detect` as multipart upload
- Users: 50 concurrent
- Duration: 60s
- Spawn rate: 5 users/sec
- Results saved to `api/locust/results/locust_YYYY-MM-DD.csv`
- Run command: `locust -f api/locust/locustfile.py --headless -u 50 -r 5 -t 60s --csv api/locust/results/locust_$(date +%F)`

---

## Error Handling

| Scenario | Response |
|---|---|
| Invalid image bytes | 422 Unprocessable Entity |
| URL fetch fails / non-image | 400 Bad Request |
| Triton unavailable | 503 Service Unavailable |
| Batcher queue full (>500 items) | 429 Too Many Requests |

---

## Testing

Unit tests (no Triton required):
- `test_preprocess.py`: shape, dtype, normalization, URL load (mocked httpx)
- `test_batcher.py`: single submit resolves, batch collects N items, queue-full raises

Integration test (Triton must be running):
- `test_infer.py` (existing in `serving/tests/`) — already passing

---

## COCO Class Labels

`api/labels.py` — 80-class COCO list indexed 0–79, used to map class index → name in detections.
