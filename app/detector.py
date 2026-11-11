"""Optional YOLO lesion detector (Ultralytics). Train on PlantDoc / your own labelled leaves."""
import os


class LesionDetector:
    def __init__(self, weights: str):
        self.model = None
        if os.path.exists(weights):
            from ultralytics import YOLO
            self.model = YOLO(weights)

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def detect(self, bgr, conf: float = 0.35) -> list[dict]:
        if not self.enabled:
            return []
        res = self.model.predict(bgr, conf=conf, verbose=False)[0]
        return [{"label": res.names[int(c)], "confidence": float(s), "box": b.tolist()}
                for b, s, c in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls)]
