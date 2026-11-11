"""Train a YOLO detector for diseased leaves / lesions and drop the weights into models/.
Usage (Colab GPU):
  python training/train_yolo.py --data training/data.yaml --model yolov8s.pt --epochs 100
Then restart the API: it loads models/yolo_lesion.pt automatically.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="training/data.yaml")
ap.add_argument("--model", default="yolov8s.pt", help="yolov8n/s/m.pt or yolo11n/s/m.pt")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="0", help="0 for GPU, cpu for CPU")
ap.add_argument("--out", default="models/yolo_lesion.pt")
a = ap.parse_args()

model = YOLO(a.model)
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    patience=20, cos_lr=True, seed=1, project="runs", name="plant_yolo", exist_ok=True,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=15, fliplr=0.5, mosaic=1.0,  # field-photo augmentation
)

metrics = YOLO(str(Path(model.trainer.best))).val(data=a.data, split="val")
print(f"val mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")

Path(a.out).parent.mkdir(exist_ok=True)
shutil.copy(model.trainer.best, a.out)
print("saved ->", a.out)
