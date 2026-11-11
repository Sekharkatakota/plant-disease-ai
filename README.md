# Plant Disease AI

Pipeline: image -> OpenCV (CLAHE, denoise) -> ViT classifier -> (optional) YOLO lesion boxes
-> OpenCV severity % -> multilingual treatment advice -> FastAPI -> Streamlit UI.

## Models
Put in `models/`: `class_names.json` (included) and either `plant_disease_recog_model_pwp.keras` (Keras backend) or `vit_plant.pt` (ViT backend). Backend is auto-picked (ViT if present, else Keras); override with CLASSIFIER_BACKEND=keras|vit.

## Run
1. Train: `python training/train_vit.py --data dataset --epochs 5`  (creates models/vit_plant.pt + class_names.json)
2. (Optional) Train YOLO on labelled lesions, save as models/yolo_lesion.pt
3. API: `uvicorn app.main:app --reload`   UI: `streamlit run streamlit_app.py`
4. Docker: `docker compose up --build`
5. Tests: `pytest`

## Extend
- Add every class to data/solutions.json (en + te + hi curated; others auto-translated).
- Treatments must be reviewed by an agronomist before real deployment.

## YOLO (optional detector)
1. Get a *detection* dataset (PlantDoc, or your own photos labelled in Roboflow/LabelImg). PlantVillage has no boxes.
2. VOC xml -> YOLO format: `python training/voc_to_yolo.py --src <folder with jpg+xml> --out yolo_data` (Roboflow users: export "YOLOv8" instead).
3. Train: `python training/train_yolo.py --data training/data.yaml --model yolov8s.pt --epochs 100`
4. Weights are copied to `models/yolo_lesion.pt`; restart the API and `/health` shows `"yolo": true`.
