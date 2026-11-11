"""Classifier backends: KerasClassifier (MobileNetV2) and ViTClassifier (timm).
Both return [{"label", "confidence"}, ...] sorted by confidence."""
import json
import os

import cv2
import numpy as np


def load_names(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def _topk(probs: np.ndarray, names: list[str], k: int):
    idx = np.argsort(probs)[::-1][:k]
    return [{"label": names[i], "confidence": round(float(probs[i]), 4)} for i in idx]


class KerasClassifier:
    """Loads the model saved by the original notebook (160x160, preprocess_input inside the model)."""

    def __init__(self, model_path: str, names_path: str):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        self.names = load_names(names_path)
        self.size = tuple(self.model.input_shape[1:3]) or (160, 160)

    def predict(self, bgr: np.ndarray, k: int = 3):
        rgb = cv2.cvtColor(cv2.resize(bgr, self.size), cv2.COLOR_BGR2RGB).astype("float32")
        p = self.model.predict(rgb[None], verbose=0)[0].astype("float64")
        # The notebook used sigmoid + SparseCategoricalCrossentropy, so outputs may not sum to 1.
        # Normalising matches what that loss optimised and makes confidences meaningful.
        p = p / max(p.sum(), 1e-9)
        return _topk(p, self.names, k)


class ViTClassifier:
    def __init__(self, weights: str, names_path: str, device: str | None = None):
        import timm, torch
        self.torch = torch
        self.names = load_names(names_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                       num_classes=len(self.names))
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.to(self.device).eval()

    def predict(self, bgr: np.ndarray, k: int = 3):
        torch = self.torch
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5  # timm ViT mean/std = 0.5
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return _topk(probs, self.names, k)


def build_classifier(cfg):
    backend = cfg.CLASSIFIER_BACKEND
    if backend == "auto":
        backend = "vit" if os.path.exists(cfg.VIT_WEIGHTS) else "keras"
    if backend == "vit":
        return ViTClassifier(cfg.VIT_WEIGHTS, cfg.CLASS_NAMES)
    return KerasClassifier(cfg.KERAS_MODEL, cfg.CLASS_NAMES)
