import base64
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import config, vision
from .classifier import build_classifier
from .detector import LesionDetector
from .knowledge import Knowledge

log = logging.getLogger("plant-ai")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = build_classifier(config)
    state["det"] = LesionDetector(config.YOLO_WEIGHTS)
    state["kb"] = Knowledge(config.SOLUTIONS)
    log.info("models loaded (yolo=%s)", state["det"].enabled)
    yield
    state.clear()


app = FastAPI(title="Plant Disease AI", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "yolo": state["det"].enabled, "classifier": type(state["clf"]).__name__}


@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form("en")):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Upload a JPG/PNG/WEBP image")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "Image too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    img = vision.enhance(img)
    top = state["clf"].predict(img, k=3)
    boxes = state["det"].detect(img)
    sev = vision.severity(img)
    best = top[0]
    uncertain = best["confidence"] < config.CONF_THRESHOLD

    annotated = vision.draw_boxes(img, boxes)
    ok, buf = cv2.imencode(".jpg", annotated)
    return {
        "prediction": best,
        "top3": top,
        "uncertain": uncertain,
        "severity": sev,
        "detections": boxes,
        "advice": None if uncertain else state["kb"].advice(best["label"], lang),
        "lang": lang,
        "annotated_image": base64.b64encode(buf).decode() if ok else None,
        "disclaimer": "AI-assisted result. Confirm with a local agriculture expert before spraying.",
    }
