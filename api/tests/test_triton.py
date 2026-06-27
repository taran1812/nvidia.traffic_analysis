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
    x1, y1, x2, y2 = result[0][0].bbox
    assert x1 == pytest.approx(270/640, abs=0.001)
    assert y1 == pytest.approx(270/640, abs=0.001)
    assert x2 == pytest.approx(370/640, abs=0.001)
    assert y2 == pytest.approx(370/640, abs=0.001)


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
    assert x1 == pytest.approx(0.0, abs=0.001)
    assert y1 == pytest.approx(0.0, abs=0.001)
    assert x2 == pytest.approx(1.0, abs=0.001)
    assert y2 == pytest.approx(1.0, abs=0.001)


def test_parse_output_filters_low_confidence():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    raw[0, 0, 0] = 320.0
    raw[0, 1, 0] = 320.0
    raw[0, 2, 0] = 100.0
    raw[0, 3, 0] = 100.0
    raw[0, 4, 0] = 0.1  # below threshold
    result = parse_output(raw, conf_thresh=0.25, iou_thresh=0.45)
    assert result == [[]]
