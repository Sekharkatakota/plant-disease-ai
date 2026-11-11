"""Fine-tune ViT-B/16 on the PlantVillage folder split (train/val/test via splitfolders).
Usage: python training/train_vit.py --data dataset --epochs 5
"""
import argparse, json
import timm, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-5)
a = ap.parse_args()

norm = T.Normalize([0.5] * 3, [0.5] * 3)
tr_tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
ev_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
tr = datasets.ImageFolder(f"{a.data}/train", tr_tf)
va = datasets.ImageFolder(f"{a.data}/val", ev_tf)
te = datasets.ImageFolder(f"{a.data}/test", ev_tf)
dl = lambda ds, sh: DataLoader(ds, a.bs, shuffle=sh, num_workers=2, pin_memory=True)
tr_dl, va_dl, te_dl = dl(tr, True), dl(va, False), dl(te, False)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=len(tr.classes)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


def evaluate(loader):
    model.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            ok += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return ok / n


json.dump(tr.classes, open("models/class_names.json", "w"))
best = 0
for ep in range(a.epochs):
    model.train()
    for x, y in tr_dl:
        opt.zero_grad()
        crit(model(x.to(dev)), y.to(dev)).backward()
        opt.step()
    acc = evaluate(va_dl)
    print(f"epoch {ep+1}: val_acc={acc:.4f}")
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "models/vit_plant.pt")
model.load_state_dict(torch.load("models/vit_plant.pt"))
print("test_acc:", evaluate(te_dl))
