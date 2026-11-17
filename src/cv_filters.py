import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import CLIPModel, CLIPProcessor


class BurstPreFilter:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        model_id = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

    def analyze_technical_quality(self, image_path: str) -> dict:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        global_sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        eye_sharpness_scores = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
            for (ex, ey, ew, eh) in eyes:
                eye_patch = roi_gray[ey : ey + eh, ex : ex + ew]
                eye_sharpness_scores.append(float(cv2.Laplacian(eye_patch, cv2.CV_64F).var()))

        eye_sharpness = float(np.mean(eye_sharpness_scores)) if eye_sharpness_scores else global_sharpness
        blown_ratio = float(np.mean(gray > 250))
        crushed_ratio = float(np.mean(gray < 5))
        luma_drift = max(0.0, (abs(float(gray.mean()) - 120.0) - 35.0) / 120.0)
        exposure_penalty = min(1.0, (blown_ratio + crushed_ratio) * 2.5 + luma_drift)

        return {
            "global_sharpness": round(global_sharpness, 2),
            "eye_sharpness": round(eye_sharpness, 2),
            "faces_detected": int(len(faces)),
            "exposure_penalty": round(exposure_penalty, 3),
        }

    @torch.inference_mode()
    def cluster_burst_shots(self, image_paths: list[str], distance_threshold: float = 0.22) -> list[int]:
        if len(image_paths) <= 1:
            return [0] * len(image_paths)

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model.get_image_features(**inputs)

        embeddings = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1).cpu().numpy()

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        return clustering.fit_predict(embeddings).tolist()
