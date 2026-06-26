import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_counter import VehicleCounter


def test_per_frame_counts_cars():
    counter = VehicleCounter(fps=30.0)
    result = counter.update(frame_num=0, detections=[2, 2, 2])
    assert result['frame']['car'] == 3
    assert result['frame']['truck'] == 0
    assert result['frame']['bus'] == 0
    assert result['frame']['motorcycle'] == 0


def test_per_frame_counts_mixed():
    counter = VehicleCounter(fps=30.0)
    result = counter.update(frame_num=0, detections=[2, 7, 5, 3])
    assert result['frame']['car'] == 1
    assert result['frame']['truck'] == 1
    assert result['frame']['bus'] == 1
    assert result['frame']['motorcycle'] == 1


def test_unknown_class_ignored():
    counter = VehicleCounter(fps=30.0)
    result = counter.update(frame_num=0, detections=[0, 1, 6, 15])
    assert sum(result['frame'].values()) == 0
    assert sum(result['per_minute'].values()) == 0


def test_per_minute_accumulates():
    counter = VehicleCounter(fps=30.0)
    counter.update(frame_num=0, detections=[2])
    result = counter.update(frame_num=15, detections=[2])
    assert result['per_minute']['car'] == 2


def test_per_minute_sliding_window():
    counter = VehicleCounter(fps=30.0)
    # frame 0: 1 car — will be at t=0s
    counter.update(frame_num=0, detections=[2])
    # frame 1800: t=60s — frame 0 is exactly at the boundary (cutoff = 1800 - 1800 = 0)
    # frame 0 has frame_num=0 which is NOT < 0, so it stays
    result = counter.update(frame_num=1800, detections=[])
    assert result['per_minute']['car'] == 1
    # frame 1801: cutoff = 1, frame 0 drops out
    result = counter.update(frame_num=1801, detections=[])
    assert result['per_minute']['car'] == 0


def test_return_keys_always_present():
    counter = VehicleCounter(fps=30.0)
    result = counter.update(frame_num=0, detections=[])
    assert set(result['frame'].keys()) == {'car', 'truck', 'bus', 'motorcycle'}
    assert set(result['per_minute'].keys()) == {'car', 'truck', 'bus', 'motorcycle'}
