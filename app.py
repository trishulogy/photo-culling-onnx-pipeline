import os
import tempfile
import onnxruntime as ort
import streamlit as st
from PIL import Image
from src.cv_filters import BurstPreFilter
from src.aesthetic_model import get_eval_transform
from src.export_onnx import export_and_benchmark

st.set_page_config(page_title="AI Photo Culling & Quality Scorer", layout="wide")
st.title("AI Photo Culling & Edge Quality Scorer (PyTorch + CLIP + ONNX)")

ONNX_PATH = "models/aesthetic_scorer.onnx"
if not os.path.exists(ONNX_PATH):
    with st.spinner("Exporting MobileNetV3 Aesthetic Scorer to ONNX..."):
        export_and_benchmark(ONNX_PATH, runs=30)


@st.cache_resource
def load_engines():
    pre_filter = BurstPreFilter()
    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return pre_filter, ort_session


pre_filter, ort_session = load_engines()
transform = get_eval_transform()

uploaded_files = st.file_uploader(
    "Upload a burst of photos (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for f in uploaded_files:
            p = os.path.join(tmpdir, f.name)
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            paths.append(p)

        clusters = pre_filter.cluster_burst_shots(paths, distance_threshold=0.22)
        records = []

        for path, cluster_id in zip(paths, clusters):
            tech = pre_filter.analyze_technical_quality(path)
            img_tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).numpy()
            aesthetic_score = float(ort_session.run(None, {"image": img_tensor})[0][0])

            focus_norm = min(10.0, tech["eye_sharpness"] / 50.0)
            composite = round(
                (0.45 * aesthetic_score) + (0.45 * focus_norm) - (2.0 * tech["exposure_penalty"]),
                2,
            )
            records.append({
                "name": os.path.basename(path),
                "path": path,
                "cluster": cluster_id,
                "composite": composite,
                "aesthetic": round(aesthetic_score, 2),
                **tech,
            })

        for c_id in set(clusters):
            group = [r for r in records if r["cluster"] == c_id]
            best = max(group, key=lambda x: x["composite"])
            for r in group:
                r["status"] = "KEEPER" if r["name"] == best["name"] else "DUPLICATE / REJECT"

        for c_id in sorted(set(clusters)):
            st.subheader(f"Burst Cluster #{c_id + 1}")
            group = sorted([r for r in records if r["cluster"] == c_id], key=lambda x: -x["composite"])
            cols = st.columns(len(group))
            for col, item in zip(cols, group):
                with col:
                    st.image(item["path"], use_container_width=True)
                    badge = "🟢 KEEPER" if item["status"] == "KEEPER" else "⚪ REJECT"
                    st.markdown(f"**{badge}** — `{item['name']}`")
                    st.caption(
                        f"Score: **{item['composite']}/10** | Aesthetic: {item['aesthetic']} | "
                        f"Eye Focus: {item['eye_sharpness']} | Faces: {item['faces_detected']}"
                    )
