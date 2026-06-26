import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from benchmark import preprocess_frame, preprocess_image_file, compute_stats


def test_preprocess_frame_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.shape == (3, 640, 640)


def test_preprocess_frame_dtype():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.dtype == np.float32


def test_preprocess_frame_normalized():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_preprocess_image_file_output_shape():
    img_path = os.path.join(os.path.dirname(__file__), '../../bus.jpg')
    if not os.path.exists(img_path):
        pytest.skip("bus.jpg not found")
    frames = preprocess_image_file(img_path, n=10)
    assert frames.shape == (10, 3, 640, 640)
    assert frames.dtype == np.float32


def test_compute_stats_percentiles():
    latencies = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_stats(latencies, batch_size=4)
    assert stats['batch_size'] == 4
    assert stats['mean_latency_ms'] == pytest.approx(30.0, abs=0.1)
    assert stats['p50_latency_ms'] == pytest.approx(30.0, abs=0.1)
    assert stats['p95_latency_ms'] == pytest.approx(48.0, abs=1.0)
    assert stats['throughput_fps'] == pytest.approx(4 * 1000 / 30.0, abs=0.1)


def test_compute_stats_throughput_scales_with_batch():
    latencies = np.array([10.0] * 100)
    stats1 = compute_stats(latencies, batch_size=1)
    stats4 = compute_stats(latencies, batch_size=4)
    assert stats4['throughput_fps'] == pytest.approx(4 * stats1['throughput_fps'], abs=0.1)
