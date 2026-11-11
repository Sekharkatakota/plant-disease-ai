"""Convert Pascal-VOC annotations (image + .xml pairs in one folder, e.g. PlantDoc) to a YOLO dataset.

Usage:
  python training/voc_to_yolo.py --src PlantDoc/TRAIN --out yolo_data
  python training/voc_to_yolo.py --src my_labelled --out yolo_data --single-class leaf_disease
Creates yolo_data/{images,labels}/{train,val,test} and training/data.yaml
"""
import argparse, random, shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder with images and VOC .xml files")
ap.add_argument("--out", default="yolo_data")
ap.add_argument("--val", type=float, default=0.15)
ap.add_argument("--test", type=float, default=0.05)
ap.add_argument("--single-class", default=None, help="merge all boxes into one class with this name")
ap.add_argument("--seed", type=int, default=1337)
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
pairs = []
for xml in sorted(src.rglob("*.xml")):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        img = xml.with_suffix(ext)
        if img.exists():
            pairs.append((img, xml)); break
print("found", len(pairs), "image/xml pairs")

# class list
classes = [a.single_class] if a.single_class else sorted({
    o.findtext("name").strip() for _, x in pairs for o in ET.parse(x).getroot().iter("object")})
cid = {c: i for i, c in enumerate(classes)}


def to_yolo(xml: Path):
    r = ET.parse(xml).getroot()
    W, H = float(r.findtext("size/width")), float(r.findtext("size/height"))
    lines = []
    for o in r.iter("object"):
        name = a.single_class or o.findtext("name").strip()
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if W <= 0 or H <= 0 or x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{cid[name]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    return lines


random.Random(a.seed).shuffle(pairs)
n = len(pairs); n_test = int(n * a.test); n_val = int(n * a.val)
split = {"test": pairs[:n_test], "val": pairs[n_test:n_test + n_val], "train": pairs[n_test + n_val:]}
for name, items in split.items():
    (out / "images" / name).mkdir(parents=True, exist_ok=True)
    (out / "labels" / name).mkdir(parents=True, exist_ok=True)
    kept = 0
    for img, xml in items:
        lines = to_yolo(xml)
        if not lines:
            continue
        shutil.copy(img, out / "images" / name / img.name)
        (out / "labels" / name / (img.stem + ".txt")).write_text("\n".join(lines))
        kept += 1
    print(f"{name}: {kept} images")

yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
yaml += [f"  {i}: {c}" for i, c in enumerate(classes)]
Path("training/data.yaml").write_text("\n".join(yaml) + "\n")
print("classes:", len(classes), "-> training/data.yaml written")
