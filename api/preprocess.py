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
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {img.shape}")
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)
