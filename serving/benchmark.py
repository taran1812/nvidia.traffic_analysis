"""
Triton Inference Server benchmark for YOLOv8n.
Sweeps batch sizes 1/4/8/16, measures latency + throughput.

Usage:
    python3 serving/benchmark.py
    python3 serving/benchmark.py --url localhost:8000 --video path/to/video.mp4
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tritonclient.http as httpclient

MODEL_NAME = "yolov8n"
MODEL_VERSION = "1"
BATCH_SIZES = [1, 4, 8, 16]
WARMUP_ITERS = 10
BENCH_ITERS = 100


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize BGR frame to 640x640, normalize to [0,1] FP32, transpose to CHW."""
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return frame.transpose(2, 0, 1)


def preprocess_image_file(image_path: str, n: int = 200) -> np.ndarray:
    """Load one image, repeat n times. Returns (n, 3, 640, 640) FP32."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    frame = preprocess_frame(img)
    return np.stack([frame] * n)


def load_frames_from_video(video_path: str, n: int = 200) -> np.ndarray:
    """Load n frames from video, looping if needed. Returns (n, 3, 640, 640) FP32."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(preprocess_frame(frame))
    cap.release()
    return np.stack(frames[:n])


def compute_stats(latencies: np.ndarray, batch_size: int) -> dict:
    """Compute latency stats and throughput from per-batch latencies (ms)."""
    return {
        "batch_size": batch_size,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": float(batch_size * 1000 / np.mean(latencies)),
    }


def infer_batch(client: httpclient.InferenceServerClient, frames: np.ndarray) -> float:
    """Send one batch to Triton, return latency in ms."""
    inputs = [httpclient.InferInput("images", list(frames.shape), "FP32")]
    inputs[0].set_data_from_numpy(frames)
    outputs = [httpclient.InferRequestedOutput("output0")]
    t0 = time.perf_counter()
    client.infer(model_name=MODEL_NAME, model_version=MODEL_VERSION,
                 inputs=inputs, outputs=outputs)
    return (time.perf_counter() - t0) * 1000


def run_batch_benchmark(client: httpclient.InferenceServerClient,
                        frames: np.ndarray, batch_size: int) -> dict:
    for _ in range(WARMUP_ITERS):
        infer_batch(client, frames[:batch_size])
    latencies = []
    n = len(frames)
    for i in range(BENCH_ITERS):
        start = (i * batch_size) % max(1, n - batch_size)
        batch = frames[start:start + batch_size]
        latencies.append(infer_batch(client, batch))
    return compute_stats(np.array(latencies), batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.video and Path(args.video).exists():
        print(f"Loading frames from video: {args.video}")
        frames = load_frames_from_video(args.video)
    else:
        image_path = str(project_root / "bus.jpg")
        print(f"Loading frames from image: {image_path}")
        frames = preprocess_image_file(image_path)

    print(f"Frames: {frames.shape}  dtype={frames.dtype}\n")

    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton not ready at {args.url}")
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' not ready")
    print(f"Triton ready. Model '{MODEL_NAME}' loaded.\n")

    results = []
    for batch_size in BATCH_SIZES:
        print(f"Benchmarking batch={batch_size}...", end=" ", flush=True)
        stats = run_batch_benchmark(client, frames, batch_size)
        results.append(stats)
        print(f"mean={stats['mean_latency_ms']:.1f}ms  "
              f"p50={stats['p50_latency_ms']:.1f}ms  "
              f"p95={stats['p95_latency_ms']:.1f}ms  "
              f"fps={stats['throughput_fps']:.1f}")

    print("\n--- Benchmark Results ---")
    print(f"{'Batch':>6} {'Mean(ms)':>10} {'p50(ms)':>8} {'p95(ms)':>8} {'FPS':>8}")
    for r in results:
        print(f"{r['batch_size']:>6} {r['mean_latency_ms']:>10.1f} "
              f"{r['p50_latency_ms']:>8.1f} {r['p95_latency_ms']:>8.1f} "
              f"{r['throughput_fps']:>8.1f}")

    out_dir = script_dir / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"benchmark_{datetime.now().strftime('%Y-%m-%d')}.json"
    payload = {"date": datetime.now().isoformat(), "model": MODEL_NAME,
                "triton_url": args.url, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
