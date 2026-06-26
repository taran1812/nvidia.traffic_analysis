from collections import defaultdict, deque

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}
_LABELS = list(dict.fromkeys(VEHICLE_CLASSES.values()))


class VehicleCounter:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._window: deque = deque()  # (frame_num, label)

    def update(self, frame_num: int, detections: list) -> dict:
        frame_counts = defaultdict(int)
        for class_id in detections:
            if class_id in VEHICLE_CLASSES:
                label = VEHICLE_CLASSES[class_id]
                frame_counts[label] += 1
                self._window.append((frame_num, label))

        cutoff = frame_num - int(self.fps * 60)
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        per_minute = defaultdict(int)
        for _, label in self._window:
            per_minute[label] += 1

        return {
            'frame': {l: frame_counts.get(l, 0) for l in _LABELS},
            'per_minute': {l: per_minute.get(l, 0) for l in _LABELS},
        }
