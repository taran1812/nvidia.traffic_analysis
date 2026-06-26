# DeepStream Pipeline Design — Week 2

**Date:** 2026-06-25  
**Scope:** GPU-accelerated vehicle detection + counting pipeline using Nvidia DeepStream 7.1 + TensorRT FP16 YOLOv8n. Produces annotated video with bounding boxes and rolling vehicle counts.

---

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Pipeline approach | Python pyds bindings | Keeps counting logic in Python; no C compilation required |
| Output | Annotated `.mp4` | Visual proof of pipeline; demo-ready for README/LinkedIn |
| Vehicle classes | car, truck, bus, motorcycle (COCO 2,3,5,7) | Meaningful traffic signal; bicycles noisy at traffic-cam angles |
| Docker image | `nvcr.io/nvidia/deepstream:7.1-triton-multiarch` | TRT 10+ required for engine built with `execute_async_v3` API |
| YOLOv8 post-processing | Python probe callback | Avoids custom C bbox parser; keeps everything in Python for Week 2 |

---

## File Structure

```
pipeline/
├── pipeline.py              # GStreamer pipeline entry point (pyds)
├── vehicle_counter.py       # Rolling per-frame + per-minute count logic
├── config/
│   ├── nvinfer_config.txt   # nvinfer element config → yolov8n.engine
│   └── labels.txt           # COCO 80-class label list
├── output/                  # Annotated .mp4 written here (gitignored)
├── tests/
│   └── test_counter.py      # Unit tests for VehicleCounter logic
└── download_dataset.sh      # Pulls one UA-DETRAC clip for validation
Dockerfile.deepstream        # Extends deepstream:7.1-triton-multiarch
```

---

## Architecture

### GStreamer Element Chain

```
filesrc → decodebin → nvstreammux → nvinfer → nvtracker → nvdsosd → nvvideoconvert → filesink
                                        ↑
                               yolov8n.engine (TRT FP16)
                               mounted at /models/ in container
```

### Probe Callback (nvinfer src pad, per frame)

```
frame fires
  → iterate NvDsFrameMeta → NvDsObjectMeta list
  → filter: class_id in {2, 3, 5, 7}
  → VehicleCounter.update(frame_num, detections)
  → get {car, truck, bus, motorcycle} counts (per-frame + per-minute)
  → write count string to NvDsDisplayMeta
  → nvdsosd renders overlay text onto frame
```

---

## Components

### `pipeline.py`
- Builds and links GStreamer elements
- Registers probe on `nvinfer` src pad
- Handles bus messages: `GST_MESSAGE_ERROR` and `GST_MESSAGE_EOS` both trigger clean shutdown
- CLI: `python3 pipeline.py --input <video> --output <video> --engine <path>`

### `vehicle_counter.py`
- `VehicleCounter` class — pure Python, no GStreamer dependency
- `update(frame_num: int, detections: list[int]) -> dict` — returns per-frame counts + per-minute rolling totals
- Per-minute window: sliding 60s calculated from `frame_num / fps`
- Class ID → label map: `{2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}`

### `config/nvinfer_config.txt`
- Sets `model-engine-file` to mounted engine path
- `num-detected-classes=80` (COCO)
- `network-type=0` (detector)
- `interval=0` (infer every frame)

### `Dockerfile.deepstream`
```dockerfile
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch
COPY pipeline/ /app/pipeline/
WORKDIR /app
RUN pip install numpy opencv-python
# Engine mounted at runtime: -v $(pwd)/yolov8n.engine:/models/yolov8n.engine
```

---

## Error Handling

- Missing engine file or bad input path → caught at startup, fast fail with clear message before pipeline builds
- Probe callback → wrapped in try/except; bad frame metadata skips silently, never crashes pipeline
- Bus error → logs GStreamer error message, triggers clean shutdown
- No retry logic — Week 2 is single-file validation

---

## Testing + Validation

### Unit tests (`pipeline/tests/test_counter.py`)
- `test_per_frame_counts` — single frame, 3 cars detected → counts match
- `test_per_minute_window` — simulate 1800 frames at 30fps → minute boundary resets
- `test_unknown_class_ignored` — class_id=0 (person) → not counted

### Integration validation checklist (manual, run on UA-DETRAC clip)
- [ ] Pipeline runs to EOS without crash
- [ ] Output `.mp4` plays in VLC
- [ ] Bounding boxes visible on vehicles
- [ ] Overlay text shows non-zero counts
- [ ] Per-minute totals increment as video progresses

---

## Dataset

UA-DETRAC: public traffic dataset, free, no login.  
`download_dataset.sh` pulls one training clip (~100MB) for pipeline validation.  
Full dataset: ~1.5GB, not committed to repo.

---

## Out of Scope (Week 2)

- RTSP / live stream input (Week 3+)
- Triton serving (Week 3)
- Prometheus metrics (Week 5)
- Multi-stream / batched inference
