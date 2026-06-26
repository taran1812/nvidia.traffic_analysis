import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock
from api.preprocess import load_from_bytes, load_from_url, preprocess


def test_preprocess_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess(frame)
    assert result.shape == (3, 640, 640)


def test_preprocess_dtype():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = preprocess(frame)
    assert result.dtype == np.float32


def test_preprocess_bgr_to_rgb():
    # B=100, G=0, R=200 in BGR — after preprocess, channel 0 (R) ≈ 200/255, channel 2 (B) ≈ 100/255
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 100  # B channel
    frame[:, :, 2] = 200  # R channel
    result = preprocess(frame)
    assert result.max() <= 1.0
    assert result.min() >= 0.0
    assert abs(result[0].mean() - 200 / 255.0) < 0.01  # R becomes channel 0
    assert abs(result[2].mean() - 100 / 255.0) < 0.01  # B becomes channel 2


def test_preprocess_invalid_channels_raises():
    gray = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 3-channel"):
        preprocess(gray)


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


def test_load_from_url_success():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    mock_response = MagicMock()
    mock_response.content = bytes(buf)
    mock_response.raise_for_status = MagicMock()
    with patch('api.preprocess.httpx.get', return_value=mock_response) as mock_get:
        result = load_from_url('http://example.com/img.jpg')
        mock_get.assert_called_once_with('http://example.com/img.jpg', timeout=10.0, follow_redirects=True)
    assert result.shape == (100, 100, 3)


def test_load_from_url_http_error():
    import httpx
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )
    with patch('api.preprocess.httpx.get', return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            load_from_url('http://example.com/missing.jpg')
