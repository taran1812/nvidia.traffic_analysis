import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import cv2
import pytest
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
