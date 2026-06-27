from pydantic import AnyHttpUrl, BaseModel


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
    url: AnyHttpUrl


class FrameDetection(BaseModel):
    frame: int
    detections: list[Detection]


class VideoDetectResponse(BaseModel):
    frames: list[FrameDetection]
    total_frames: int
    processed_frames: int
    model: str
