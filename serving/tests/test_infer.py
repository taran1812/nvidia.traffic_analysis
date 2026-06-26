import numpy as np
import tritonclient.http as httpclient


def test_yolov8n_infer():
    client = httpclient.InferenceServerClient("localhost:8000")

    assert client.is_server_ready(), "Triton server not ready"
    assert client.is_model_ready("yolov8n"), "yolov8n model not ready"

    img = np.random.rand(1, 3, 640, 640).astype(np.float32)
    inp = httpclient.InferInput("images", img.shape, "FP32")
    inp.set_data_from_numpy(img)
    out = httpclient.InferRequestedOutput("output0")

    result = client.infer("yolov8n", inputs=[inp], outputs=[out])
    output = result.as_numpy("output0")

    assert output.shape == (1, 84, 8400), f"Unexpected output shape: {output.shape}"
    print(f"Input:  {img.shape}")
    print(f"Output: {output.shape}")
    print("INFERENCE OK")


if __name__ == "__main__":
    test_yolov8n_infer()
