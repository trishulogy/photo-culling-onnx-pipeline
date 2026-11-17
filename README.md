# AI Photo Culling & Edge Quality Scorer (PyTorch + CLIP + ONNX)

An end-to-end automated photo culling pipeline designed for local, low-latency burst photography triage. Combines **OpenCV** optical focus/exposure heuristics, **Hugging Face CLIP** perceptual duplicate clustering, and a fine-tuned **MobileNetV3-Small** aesthetic quality regressor exported to **ONNX Runtime** for edge CPU deployment.

## Architecture & Pipeline Stages
1. **Optical & Portrait Pre-Filtering (`src/cv_filters.py`):** Computes global Laplacian focus variance, Haar-cascade facial/eye-region sharpness, and highlight/shadow clipping penalties (`>250` blown highlights, `<5` crushed blacks, plus mid-tone luminance drift).
2. **Perceptual Burst Clustering (`src/cv_filters.py`):** Extracts L2-normalized embeddings via `openai/clip-vit-base-patch32` and groups burst duplicates using Agglomerative Cosine Clustering (`distance_threshold=0.22`).
3. **Aesthetic & Exposure Regressor (`src/aesthetic_model.py`):** Lightweight `MobileNetV3-Small` regression head trained with `SmoothL1Loss` on sharp vs. motion-blurred and exposure-degraded pairs to score visual quality (`0–10`).
4. **Edge Inference Optimization (`src/export_onnx.py`):** Exports the PyTorch model to ONNX (`opset 17`) with constant folding and dynamic batch axes for fast local CPU execution.

## Measured Benchmarks & Evaluation Results

| Metric | PyTorch FP32 (CPU) | ONNX Runtime (CPU) | Improvement / Result |
| :--- | :---: | :---: | :---: |
| **Single-Image Inference Latency (100 runs)** | `12.69 ms` | **`3.52 ms`** | **3.61x Speedup** |
| **Sharp vs. Degraded Score Separation** | — | **`8.12 / 10` vs `4.11 / 10`** | **`+4.01` point margin** |
| **Burst Keeper Selection (Top-1 Accuracy)** | — | **100% (3/3 scenes)** | Tuned CLIP cosine threshold (`0.15 -> 0.22`) |

## Repository Structure
```text
photo-culling-onnx-pipeline/
├── src/
│   ├── __init__.py
│   ├── cv_filters.py       # OpenCV Laplacian/Haar focus + CLIP cosine burst clustering
│   ├── aesthetic_model.py  # MobileNetV3-Small aesthetic quality regressor
│   └── export_onnx.py      # ONNX export + CPU latency benchmark
├── models/
│   └── aesthetic_scorer.onnx
├── app.py                  # Streamlit burst-photo culling dashboard
├── requirements.txt
└── README.md
