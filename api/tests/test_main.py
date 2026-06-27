import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import cv2


def make_jpeg_bytes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [100, 150, 200]
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


@pytest.fixture
def mock_batcher():
    batcher = MagicMock()
    batcher.submit = AsyncMock(return_value=([], 5.0))
    return batcher


@pytest.fixture
def client(mock_batcher):
    with patch('api.main.AsyncBatcher') as mock_batcher_cls, \
         patch('api.main.TritonClient') as mock_triton_cls:
        mock_triton_cls.return_value.is_ready.return_value = True
        mock_batcher_cls.return_value = mock_batcher
        mock_batcher_cls.return_value.start = AsyncMock()
        mock_batcher_cls.return_value.stop = AsyncMock()
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "triton" in data


def test_detect_file_upload(client, mock_batcher):
    jpeg = make_jpeg_bytes()
    resp = client.post("/detect", files={"file": ("test.jpg", jpeg, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "detections" in data
    assert "inference_time_ms" in data
    assert "model" in data
    assert "image_size" in data
    assert mock_batcher.submit.called


def test_detect_invalid_image_returns_422(client):
    resp = client.post("/detect", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 422


def test_metrics_endpoint_exists(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_health_triton_unavailable():
    with patch('api.main.AsyncBatcher') as mock_batcher_cls, \
         patch('api.main.TritonClient') as mock_triton_cls:
        mock_triton_cls.return_value.is_ready.return_value = False
        mock_batcher_cls.return_value.start = AsyncMock()
        mock_batcher_cls.return_value.stop = AsyncMock()
        from api.main import app
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            assert resp.json()["triton"] == "unavailable"


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
