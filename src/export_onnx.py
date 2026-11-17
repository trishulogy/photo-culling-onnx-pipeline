import os
import time
import onnx
import onnxruntime as ort
import torch
from src.aesthetic_model import AestheticQualityNet


def export_and_benchmark(onnx_path: str = "models/aesthetic_scorer.onnx", runs: int = 100) -> dict:
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = AestheticQualityNet(pretrained=True).eval().cpu()
    if os.path.exists("models/aesthetic_scorer.pth"):
        model.load_state_dict(torch.load("models/aesthetic_scorer.pth", map_location="cpu"))

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["aesthetic_score"],
        dynamic_axes={"image": {0: "batch_size"}, "aesthetic_score": {0: "batch_size"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))

    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        pytorch_ms = ((time.perf_counter() - t0) / runs) * 1000

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    np_input = dummy_input.numpy()

    for _ in range(10):
        _ = session.run(None, {"image": np_input})
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = session.run(None, {"image": np_input})
    onnx_ms = ((time.perf_counter() - t0) / runs) * 1000

    speedup = round(pytorch_ms / onnx_ms, 2)
    results = {
        "pytorch_cpu_ms": round(pytorch_ms, 2),
        "onnx_cpu_ms": round(onnx_ms, 2),
        "speedup_x": speedup,
    }
    print(f"Benchmark ({runs} runs): PyTorch={results['pytorch_cpu_ms']} ms | ONNX={results['onnx_cpu_ms']} ms ({speedup}x faster)")
    return results


if __name__ == "__main__":
    export_and_benchmark()
