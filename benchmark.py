"""
3-way inference benchmark: PyTorch FP32 vs ONNX FP32 vs TensorRT FP16
Hardware: NVIDIA RTX 3050 Ti Laptop GPU
Model: YOLOv8n · 640x640
"""

import time
import numpy as np
import torch
import cv2

RUNS = 50
IMG_PATH = "bus.jpg"
ONNX_PATH = "yolov8n.onnx"
ENGINE_PATH = "yolov8n.engine"
MODEL_PATH = "yolov8n.pt"
INPUT_SIZE = (640, 640)


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def benchmark_pytorch(img_np):
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    model.model.eval().cuda()
    img_tensor = torch.from_numpy(img_np).cuda()

    for _ in range(5):
        with torch.no_grad():
            model.model(img_tensor)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.model(img_tensor)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return latencies


def benchmark_onnx(img_np):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        ONNX_PATH,
        sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    for _ in range(5):
        session.run(None, {input_name: img_np})

    latencies = []
    for _ in range(RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: img_np})
        latencies.append((time.perf_counter() - start) * 1000)

    return latencies


def benchmark_tensorrt(img_np):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(ENGINE_PATH, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # resolve I/O shapes from engine at runtime — no hardcoded assumptions
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))

    h_input = cuda.pagelocked_empty(int(np.prod(input_shape)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    np.copyto(h_input, img_np.ravel())

    def infer():
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    for _ in range(5):
        infer()

    latencies = []
    for _ in range(RUNS):
        start = time.perf_counter()
        infer()
        latencies.append((time.perf_counter() - start) * 1000)

    return latencies


def report(name, latencies):
    arr = np.array(latencies)
    mean_ms = arr.mean()
    fps = 1000.0 / mean_ms
    p95_ms = np.percentile(arr, 95)
    print(f"{name:<28} {mean_ms:>8.1f}ms   {fps:>7.1f} FPS   p95={p95_ms:.1f}ms")


def main():
    print(f"Benchmark: YOLOv8n · {INPUT_SIZE[0]}x{INPUT_SIZE[1]} · {RUNS} runs")
    print(f"Image: {IMG_PATH}\n")
    print(f"{'Mode':<28} {'Latency':>9}   {'FPS':>11}   {'p95':>10}")
    print("-" * 65)

    img_np = preprocess(IMG_PATH)

    pt_lat = benchmark_pytorch(img_np)
    report("PyTorch FP32 (baseline)", pt_lat)

    onnx_lat = benchmark_onnx(img_np)
    report("ONNX FP32", onnx_lat)

    trt_lat = benchmark_tensorrt(img_np)
    report("TensorRT FP16", trt_lat)

    baseline = np.array(pt_lat).mean()
    trt_mean = np.array(trt_lat).mean()
    speedup = baseline / trt_mean
    latency_reduction = (1 - trt_mean / baseline) * 100

    print("-" * 65)
    print(f"\nTensorRT FP16 speedup: {speedup:.2f}x  |  latency reduction: {latency_reduction:.0f}%")


if __name__ == "__main__":
    main()
