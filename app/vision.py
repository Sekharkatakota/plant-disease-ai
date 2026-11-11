"""OpenCV utilities: enhancement, leaf segmentation, severity estimate, drawing."""
import cv2
import numpy as np


def enhance(bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)  # fix uneven light
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)


def leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Foreground (leaf) mask via Otsu on saturation + morphology."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, m = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def severity(bgr: np.ndarray) -> dict:
    """% of leaf area that is NOT healthy green (brown/yellow/dark spots)."""
    mask = leaf_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    leaf_px = int(np.count_nonzero(mask))
    if leaf_px == 0:
        return {"percent": 0.0, "level": "unknown"}
    diseased = np.count_nonzero(cv2.bitwise_and(mask, cv2.bitwise_not(green)))
    pct = round(100.0 * diseased / leaf_px, 1)
    level = "low" if pct < 10 else "moderate" if pct < 30 else "high"
    return {"percent": pct, "level": level}


def draw_boxes(bgr: np.ndarray, boxes: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f'{b["label"]} {b["confidence"]:.2f}', (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out
