import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
import cv2
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.batcher import AsyncBatcher
from api.metrics import detection_total, inference_duration_seconds
from api.preprocess import load_from_bytes, load_from_url, preprocess
from api.schemas import DetectResponse, DetectURLRequest, FrameDetection, VideoDetectResponse
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


app = FastAPI(title="Traffic Analytics API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health():
    triton_status = "ready" if (_triton and _triton.is_ready()) else "unavailable"
    return {"status": "ok", "triton": triton_status}


@app.post("/detect", response_model=DetectResponse)
async def detect_file(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = load_from_bytes(data)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid image")
    h, w = img.shape[:2]
    frame = preprocess(img)
    if _batcher is None:
        raise HTTPException(status_code=503, detail="Service not ready")
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


@app.post("/detect/url", response_model=DetectResponse)
async def detect_url(body: DetectURLRequest):
    import httpx
    try:
        img = load_from_url(str(body.url))
    except (httpx.HTTPStatusError, httpx.RequestError, ValueError, OSError):
        raise HTTPException(status_code=400, detail="Failed to fetch or decode image from URL")
    h, w = img.shape[:2]
    frame = preprocess(img)
    if _batcher is None:
        raise HTTPException(status_code=503, detail="Service not ready")
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


@app.post("/detect/video", response_model=VideoDetectResponse)
async def detect_video(file: UploadFile = File(...)):
    data = await file.read()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Invalid or unsupported video file")
        raw_frames = []
        MAX_FRAMES = 300
        while len(raw_frames) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(preprocess(frame))
        cap.release()
    finally:
        os.unlink(tmp.name)

    if not raw_frames:
        raise HTTPException(status_code=422, detail="No frames extracted from video")
    if _batcher is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    frame_detections = []
    for idx, frame in enumerate(raw_frames):
        dets, _ = await _batcher.submit(frame)
        for det in dets:
            detection_total.labels(class_name=det.class_name).inc()
        frame_detections.append(FrameDetection(frame=idx, detections=dets))

    return VideoDetectResponse(
        frames=frame_detections,
        total_frames=len(raw_frames),
        processed_frames=len(frame_detections),
        model="yolov8n",
    )
