import numpy as np, cv2
from app import vision
from app.knowledge import Knowledge


def test_severity_green_leaf_is_low():
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 70, (40, 160, 40), -1)  # green "leaf"
    assert vision.severity(img)["level"] in ("low", "unknown")


def test_knowledge_curated_and_generic():
    kb = Knowledge("data/solutions.json")
    assert kb.advice("Tomato___Late_blight", "te")["source"] == "curated"
    assert kb.advice("Unknown___class", "en")["treatment"]


def test_all_disease_classes_have_curated_te_hi():
    import json
    names = json.load(open("models/class_names.json"))
    kb = Knowledge("data/solutions.json")
    for n in names:
        if n.endswith("healthy") or n.startswith("Background"):
            continue
        for lang in ("en", "te", "hi"):
            assert kb.advice(n, lang)["source"] == "curated", (n, lang)
