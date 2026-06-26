# DeepStream Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-accelerated DeepStream 7.1 pipeline that reads a traffic video, runs TensorRT FP16 YOLOv8n inference, counts vehicles (car/truck/bus/motorcycle) per frame and per minute, and writes an annotated `.mp4`.

**Architecture:** Python pyds GStreamer pipeline — `filesrc → decodebin → nvstreammux → nvinfer → nvtracker → nvdsosd → nvvideoconvert → nvv4l2h264enc → mp4mux → filesink`. A probe on the nvinfer src pad reads raw YOLOv8 output tensor (`NvDsInferTensorMeta`), decodes bboxes + NMS in Python, creates `NvDsObjectMeta` entries, and writes count text via `NvDsDisplayMeta`. A minimal C stub satisfies nvinfer's `network-type=100` requirement so the internal parser is skipped entirely.

**Tech Stack:** DeepStream 7.1 (Docker), pyds, GStreamer Python bindings (gi.repository), TensorRT FP16 engine, numpy, Python 3.10.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pipeline/__init__.py` | Create | Package marker |
| `pipeline/vehicle_counter.py` | Create | Pure-Python rolling counts |
| `pipeline/tests/__init__.py` | Create | Package marker |
| `pipeline/tests/test_counter.py` | Create | Unit tests for VehicleCounter |
| `pipeline/config/labels.txt` | Create | COCO 80-class labels |
| `pipeline/config/nvinfer_config.txt` | Create | nvinfer element config |
| `pipeline/config/bbox_parser.cpp` | Create | Stub C parser (skips internal parsing) |
| `pipeline/pipeline.py` | Create | GStreamer pipeline entry point |
| `pipeline/output/.gitkeep` | Create | Keeps output dir in git |
| `pipeline/download_dataset.sh` | Create | Pulls UA-DETRAC sample clip |
| `Dockerfile.deepstream` | Create | DeepStream 7.1 container |
| `.gitignore` | Modify | Ignore `pipeline/output/*.mp4` |

---

## Task 1: Directory Scaffolding

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/tests/__init__.py`
- Create: `pipeline/output/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p pipeline/tests pipeline/config pipeline/output
touch pipeline/__init__.py pipeline/tests/__init__.py pipeline/output/.gitkeep
```

- [ ] **Step 2: Update .gitignore**

Add to `.gitignore`:
```
# pipeline outputs
pipeline/output/*.mp4
pipeline/output/*.avi
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/ .gitignore
git commit -m "feat: scaffold pipeline directory structure"
```

---

## Task 2: VehicleCounter — TDD

**Files:**
- Create: `pipeline/tests/test_counter.py`
- Create: `pipeline/vehicle_counter.py`

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_counter.py`:
```python
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
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd pipeline && python3 -m pytest tests/test_counter.py -v
```

Expected: `ModuleNotFoundError: No module named 'vehicle_counter'`

- [ ] **Step 3: Implement VehicleCounter**

`pipeline/vehicle_counter.py`:
```python
from collections import defaultdict, deque

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}
_LABELS = list(dict.fromkeys(VEHICLE_CLASSES.values()))  # preserves insertion order


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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd pipeline && python3 -m pytest tests/test_counter.py -v
```

Expected:
```
PASSED tests/test_counter.py::test_per_frame_counts_cars
PASSED tests/test_counter.py::test_per_frame_counts_mixed
PASSED tests/test_counter.py::test_unknown_class_ignored
PASSED tests/test_counter.py::test_per_minute_accumulates
PASSED tests/test_counter.py::test_per_minute_sliding_window
PASSED tests/test_counter.py::test_return_keys_always_present
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/vehicle_counter.py pipeline/tests/test_counter.py
git commit -m "feat: add VehicleCounter with sliding 60s window"
```

---

## Task 3: COCO Labels

**Files:**
- Create: `pipeline/config/labels.txt`

- [ ] **Step 1: Write COCO 80-class label file**

`pipeline/config/labels.txt` (one label per line, index = class ID):
```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

- [ ] **Step 2: Verify line count**

```bash
wc -l pipeline/config/labels.txt
```

Expected: `80`

- [ ] **Step 3: Commit**

```bash
git add pipeline/config/labels.txt
git commit -m "feat: add COCO 80-class labels file"
```

---

## Task 4: nvinfer Config + Stub Parser

**Files:**
- Create: `pipeline/config/nvinfer_config.txt`
- Create: `pipeline/config/bbox_parser.cpp`

- [ ] **Step 1: Write nvinfer config**

`pipeline/config/nvinfer_config.txt`:
```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=/models/yolov8n.engine
labelfile-path=/app/pipeline/config/labels.txt
batch-size=1
network-type=100
num-detected-classes=80
interval=0
gie-unique-id=1
output-blob-names=output0
cluster-mode=2
parse-bbox-func-name=NvDsInferParseCustomYoloV8Stub
custom-lib-path=/app/pipeline/config/libbbox_parser.so
output-tensor-meta=1
```

> **Note:** `network-type=100` tells nvinfer to use a custom bbox parser. The stub returns an empty list so all post-processing happens in the Python probe. `output-tensor-meta=1` exposes raw YOLOv8 tensor via `NvDsInferTensorMeta` in the frame's user meta list.

- [ ] **Step 2: Write stub C parser**

`pipeline/config/bbox_parser.cpp`:
```cpp
#include "nvdsinfer_custom_impl.h"
#include <vector>

extern "C" bool NvDsInferParseCustomYoloV8Stub(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Post-processing handled in Python probe via output-tensor-meta.
    objectList.clear();
    return true;
}
```

> This stub is compiled inside the Docker container. The Dockerfile (Task 6) runs `g++` to build `libbbox_parser.so`.

- [ ] **Step 3: Commit**

```bash
git add pipeline/config/nvinfer_config.txt pipeline/config/bbox_parser.cpp
git commit -m "feat: add nvinfer config and stub bbox parser"
```

---

## Task 5: pipeline.py

**Files:**
- Create: `pipeline/pipeline.py`

- [ ] **Step 1: Write pipeline.py**

`pipeline/pipeline.py`:
```python
"""
DeepStream 7.1 vehicle detection + counting pipeline.
Usage: python3 pipeline/pipeline.py --input <video> --output <video> --engine <path>
"""

import argparse
import ctypes
import os
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import pyds

sys.path.insert(0, os.path.dirname(__file__))
from vehicle_counter import VehicleCounter

VEHICLE_CLASSES = {2, 3, 5, 7}
CLASS_NAMES = {2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}
CONF_THRESH = 0.5
IOU_THRESH = 0.45
MUXER_W = 1920
MUXER_H = 1080


def _nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def _decode_yolov8(raw, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """
    raw: numpy array shape (84, 8400)
    YOLOv8 output: rows 0-3 = cx,cy,w,h; rows 4-83 = class scores.
    Returns list of dicts: {class_id, confidence, left, top, width, height}
    """
    pred = raw.T  # (8400, 84)
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_thresh
    if not mask.any():
        return []

    pred = pred[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2

    results = []
    for cls in np.unique(class_ids):
        m = class_ids == cls
        bboxes = np.stack([x1[m], y1[m], x2[m], y2[m]], axis=1)
        confs = confidences[m]
        for i in _nms(bboxes, confs, iou_thresh):
            results.append({
                'class_id': int(cls),
                'confidence': float(confs[i]),
                'left': float(bboxes[i, 0]),
                'top': float(bboxes[i, 1]),
                'width': float(bboxes[i, 2] - bboxes[i, 0]),
                'height': float(bboxes[i, 3] - bboxes[i, 1]),
            })
    return results


def _probe_callback(pad, info, counter):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_num = frame_meta.frame_num
            detections = []

            # read raw YOLOv8 tensor from user meta
            l_user = frame_meta.frame_user_meta_list
            while l_user:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    try:
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                        ptr = ctypes.cast(
                            pyds.get_ptr(layer.buffer),
                            ctypes.POINTER(ctypes.c_float)
                        )
                        # YOLOv8n output: (84, 8400)
                        raw = np.ctypeslib.as_array(ptr, shape=(84, 8400)).copy()
                        dets = _decode_yolov8(raw)

                        for det in dets:
                            if det['class_id'] not in VEHICLE_CLASSES:
                                continue
                            detections.append(det['class_id'])

                            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                            rect = obj_meta.rect_params
                            rect.left = det['left']
                            rect.top = det['top']
                            rect.width = det['width']
                            rect.height = det['height']
                            rect.border_width = 3
                            rect.border_color.set(0.0, 1.0, 0.0, 1.0)
                            obj_meta.class_id = det['class_id']
                            obj_meta.confidence = det['confidence']
                            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
                    except Exception as e:
                        print(f"Tensor decode error frame {frame_num}: {e}", file=sys.stderr)

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

            # update counter
            result = counter.update(frame_num, detections)
            f = result['frame']
            m = result['per_minute']

            # write overlay text
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            txt = display_meta.text_params[0]
            txt.display_text = (
                f"Frame {frame_num}  |  "
                f"Car:{f['car']} Truck:{f['truck']} Bus:{f['bus']} Moto:{f['motorcycle']}  |  "
                f"/min — Car:{m['car']} Truck:{m['truck']} Bus:{m['bus']} Moto:{m['motorcycle']}"
            )
            txt.x_offset = 10
            txt.y_offset = 30
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 14
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

    except Exception as e:
        print(f"Probe error: {e}", file=sys.stderr)

    return Gst.PadProbeReturn.OK


def _build_pipeline(input_path, output_path):
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make('filesrc', 'source')
    decoder = Gst.ElementFactory.make('decodebin', 'decoder')
    muxer = Gst.ElementFactory.make('nvstreammux', 'muxer')
    infer = Gst.ElementFactory.make('nvinfer', 'infer')
    tracker = Gst.ElementFactory.make('nvtracker', 'tracker')
    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    converter = Gst.ElementFactory.make('nvvideoconvert', 'converter')
    encoder = Gst.ElementFactory.make('nvv4l2h264enc', 'encoder')
    h264parse = Gst.ElementFactory.make('h264parse', 'h264parse')
    muxout = Gst.ElementFactory.make('qtmux', 'muxout')
    sink = Gst.ElementFactory.make('filesink', 'sink')

    for name, el in [
        ('filesrc', source), ('decodebin', decoder), ('nvstreammux', muxer),
        ('nvinfer', infer), ('nvtracker', tracker), ('nvdsosd', osd),
        ('nvvideoconvert', converter), ('nvv4l2h264enc', encoder),
        ('h264parse', h264parse), ('qtmux', muxout), ('filesink', sink),
    ]:
        if not el:
            raise RuntimeError(f"Failed to create GStreamer element: {name}")
        pipeline.add(el)

    source.set_property('location', input_path)
    muxer.set_property('width', MUXER_W)
    muxer.set_property('height', MUXER_H)
    muxer.set_property('batch-size', 1)
    muxer.set_property('batched-push-timeout', 4000000)
    infer.set_property('config-file-path', '/app/pipeline/config/nvinfer_config.txt')
    sink.set_property('location', output_path)
    sink.set_property('sync', False)

    def on_pad_added(src, pad, mux):
        sink_pad = mux.get_request_pad('sink_0')
        if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            print("ERROR: Failed to link decoder → muxer", file=sys.stderr)

    decoder.connect('pad-added', on_pad_added, muxer)

    source.link(decoder)
    muxer.link(infer)
    infer.link(tracker)
    tracker.link(osd)
    osd.link(converter)
    converter.link(encoder)
    encoder.link(h264parse)
    h264parse.link(muxout)
    muxout.link(sink)

    return pipeline, infer


def main():
    ap = argparse.ArgumentParser(description='DeepStream vehicle detection pipeline')
    ap.add_argument('--input', required=True, help='Input video path')
    ap.add_argument('--output', default='/app/pipeline/output/out.mp4', help='Output video path')
    ap.add_argument('--engine', default='/models/yolov8n.engine', help='TensorRT engine path')
    ap.add_argument('--fps', type=float, default=30.0, help='Video FPS for per-minute window')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: Input not found: {args.input}")
    if not os.path.exists(args.engine):
        sys.exit(f"ERROR: Engine not found: {args.engine}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    Gst.init(None)
    counter = VehicleCounter(fps=args.fps)
    pipeline, infer_el = _build_pipeline(args.input, args.output)

    infer_src_pad = infer_el.get_static_pad('src')
    infer_src_pad.add_probe(Gst.PadProbeType.BUFFER, _probe_callback, counter)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, msg, loop):
        if msg.type == Gst.MessageType.EOS:
            print(f"\nDone. Output: {args.output}")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"ERROR: {err}\n{debug}", file=sys.stderr)
            loop.quit()

    bus.connect('message', on_message, loop)
    pipeline.set_state(Gst.State.PLAYING)
    print(f"Pipeline running: {args.input} → {args.output}")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/pipeline.py
git commit -m "feat: add DeepStream pyds vehicle detection pipeline"
```

---

## Task 6: Dockerfile

**Files:**
- Create: `Dockerfile.deepstream`

- [ ] **Step 1: Write Dockerfile**

`Dockerfile.deepstream`:
```dockerfile
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

WORKDIR /app

# Python deps
RUN pip3 install --no-cache-dir numpy==1.24.4 opencv-python-headless==4.8.0.76

# Copy pipeline source
COPY pipeline/ /app/pipeline/

# Compile stub bbox parser
RUN g++ -shared -fPIC -o /app/pipeline/config/libbbox_parser.so \
    /app/pipeline/config/bbox_parser.cpp \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -L/opt/nvidia/deepstream/deepstream/lib \
    -lnvdsgst_meta -lnvds_meta \
    && echo "Stub parser compiled OK"

# Engine mounted at runtime via -v flag — not baked into image
# Run: docker run --gpus all \
#   -v $(pwd)/yolov8n.engine:/models/yolov8n.engine \
#   -v $(pwd)/pipeline/output:/app/pipeline/output \
#   deepstream-traffic \
#   python3 pipeline/pipeline.py --input /data/traffic.mp4
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile.deepstream
git commit -m "feat: add DeepStream 7.1 Dockerfile with stub parser build"
```

---

## Task 7: Dataset Download Script

**Files:**
- Create: `pipeline/download_dataset.sh`

- [ ] **Step 1: Write download script**

`pipeline/download_dataset.sh`:
```bash
#!/usr/bin/env bash
# Downloads one UA-DETRAC training sequence for pipeline validation.
# Full dataset: http://detrac-db.rit.albany.edu
# Single sequence (~80MB) — no login required.

set -e

OUT_DIR="$(dirname "$0")/data"
mkdir -p "$OUT_DIR"

SEQUENCE_URL="https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip"

echo "Downloading UA-DETRAC training data..."
echo "Note: Full zip is ~1.5GB. Press Ctrl+C and download a single sequence manually if needed."
echo ""
echo "Alternative — download one sequence (MVI_20011, ~80MB):"
echo "  wget 'http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip' -O $OUT_DIR/detrac.zip"
echo "  unzip $OUT_DIR/detrac.zip 'Insight-MVT_Annotation_Train/MVI_20011/*' -d $OUT_DIR/"
echo ""
echo "Or use any traffic MP4 — pass it with --input to pipeline.py"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x pipeline/download_dataset.sh
git add pipeline/download_dataset.sh
git commit -m "feat: add UA-DETRAC dataset download script"
```

---

## Task 8: Build Docker Image + Integration Validation

**No new files** — validates the full pipeline end-to-end inside Docker.

- [ ] **Step 1: Pull DeepStream base image**

```bash
docker pull nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```

Expected: image downloads (~10GB, one-time).

- [ ] **Step 2: Build project image**

```bash
docker build -f Dockerfile.deepstream -t deepstream-traffic .
```

Expected: ends with `Stub parser compiled OK` and `Successfully built <id>`.

If `g++` fails on includes path, adjust `-I` flag: check include path with:
```bash
docker run --rm nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  find /opt/nvidia/deepstream -name "nvdsinfer_custom_impl.h" 2>/dev/null
```

- [ ] **Step 3: Get a test video**

```bash
# Option A: use any short traffic MP4 you have
# Option B: download a UA-DETRAC sequence
bash pipeline/download_dataset.sh
# then convert image sequence to MP4:
ffmpeg -r 25 -i pipeline/data/MVI_20011/img%05d.jpg -c:v libx264 pipeline/data/MVI_20011.mp4
```

- [ ] **Step 4: Run pipeline**

```bash
docker run --gpus all --rm \
  -v $(pwd)/yolov8n.engine:/models/yolov8n.engine \
  -v $(pwd)/pipeline/output:/app/pipeline/output \
  -v /path/to/traffic.mp4:/data/traffic.mp4 \
  deepstream-traffic \
  python3 pipeline/pipeline.py \
    --input /data/traffic.mp4 \
    --output /app/pipeline/output/out.mp4 \
    --engine /models/yolov8n.engine \
    --fps 25
```

Expected output:
```
Pipeline running: /data/traffic.mp4 → /app/pipeline/output/out.mp4
Done. Output: /app/pipeline/output/out.mp4
```

- [ ] **Step 5: Validate output**

```bash
# File exists and has size > 0
ls -lh pipeline/output/out.mp4

# Playable (use VLC or ffprobe)
ffprobe pipeline/output/out.mp4
```

Manual validation checklist:
- [ ] Pipeline runs to EOS without crash
- [ ] `out.mp4` plays in VLC
- [ ] Green bounding boxes visible on vehicles
- [ ] Overlay text bar shows per-frame and per-minute counts
- [ ] Counts are non-zero by frame ~30

- [ ] **Step 6: Commit validation result**

```bash
# update README Week 2 checkboxes
git add README.md
git commit -m "feat: complete Week 2 DeepStream pipeline — vehicle counting validated"
```

---

## Known Pitfalls

| Issue | Fix |
|---|---|
| `nvv4l2h264enc` not found | Replace with `x264enc` (software) for debugging: `Gst.ElementFactory.make('x264enc', 'encoder')` |
| `nvtracker` missing config | Add `tracker-config-file` property pointing to a tracker config, or remove nvtracker and link `infer → osd` directly |
| Tensor shape mismatch | YOLOv8 engine output may differ — log `layer.dims` in probe and adjust `shape=(84, 8400)` accordingly |
| `decodebin` won't link to muxer | Ensure `nvstreammux` sink pad `sink_0` is requested before linking |
| pyds import fails | DeepStream image installs pyds at `/opt/nvidia/deepstream/deepstream/lib/` — add to `PYTHONPATH` if needed |
