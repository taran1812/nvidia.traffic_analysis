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

    def infer_batch(self, frames: np.ndarray) -> tuple[list[list[Detection]], float]:
        """Send [N, 3, 640, 640] FP32 batch to Triton, return (detections_per_image, elapsed_ms)."""
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
