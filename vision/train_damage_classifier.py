"""
train_damage_classifier.py
-----------------------------
Trains a small convolutional neural network, from scratch, on the
synthetic damage photos (data/damage_images/{none,minor,moderate,severe}/)
to classify damage severity - the computer vision half of this platform,
kept deliberately small (3 conv blocks, 96x96 grayscale-ish RGB inputs)
so it trains in minutes on CPU with no GPU and no pretrained-model
download, consistent with every other model choice in this portfolio
(no heavy framework where a lighter one does the job on this hardware).

Uses `torchvision`-free image loading (plain PIL + a hand-rolled Dataset)
since torchvision isn't installed on this machine and isn't needed for
something this small.

Split: stratified 80/20 train/test by class, so evaluation reflects a
genuine holdout, not training-set performance.

Output:
  vision/artifacts/damage_classifier.pt   - trained model weights
  reports/vision_evaluation.json/.md       - accuracy, confusion matrix, per-class P/R
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "damage_images"
ARTIFACTS_DIR = ROOT / "vision" / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SEED = 5
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

CLASSES = ["none", "minor", "moderate", "severe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 96
TEST_FRACTION = 0.2
BATCH_SIZE = 16
N_EPOCHS = 25
LR = 1e-3


def load_dataset():
    paths, labels = [], []
    for cls in CLASSES:
        for p in sorted((IMG_DIR / cls).glob("*.png")):
            paths.append(p)
            labels.append(CLASS_TO_IDX[cls])
    return paths, np.array(labels)


def stratified_split(paths, labels):
    train_idx, test_idx = [], []
    for cls_idx in range(len(CLASSES)):
        idx = np.where(labels == cls_idx)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * TEST_FRACTION))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def load_image_tensor(path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


class DamageCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 96 -> 48 -> 24 -> 12 after 3 pools
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def batches(paths, labels, idx, batch_size, shuffle=True):
    order = idx.copy()
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        imgs = torch.stack([load_image_tensor(paths[i]) for i in batch_idx])
        lbls = torch.tensor(labels[batch_idx], dtype=torch.long)
        yield imgs, lbls


def main():
    paths, labels = load_dataset()
    paths = np.array(paths, dtype=object)
    print(f"Loaded {len(paths)} images across {len(CLASSES)} classes: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    train_idx, test_idx = stratified_split(paths, labels)
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")

    model = DamageCNN(n_classes=len(CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss, n_correct, n_seen = 0.0, 0, 0
        for imgs, lbls in batches(paths, labels, train_idx, BATCH_SIZE):
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lbls)
            n_correct += (logits.argmax(1) == lbls).sum().item()
            n_seen += len(lbls)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{N_EPOCHS}  loss={total_loss/n_seen:.4f}  "
                  f"train_acc={n_correct/n_seen:.3f}")

    # ------------------------------------------------------------------
    # Evaluation on the held-out test split
    # ------------------------------------------------------------------
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, lbls in batches(paths, labels, test_idx, BATCH_SIZE, shuffle=False):
            logits = model(imgs)
            preds = logits.argmax(1)
            all_preds.extend(preds.tolist())
            all_true.extend(lbls.tolist())

    all_preds, all_true = np.array(all_preds), np.array(all_true)
    accuracy = float((all_preds == all_true).mean())

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(all_true, all_preds):
        confusion[t, p] += 1

    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[cls] = {"precision": round(precision, 3), "recall": round(recall, 3),
                            "support": int(confusion[i, :].sum())}

    print(f"\nTest accuracy: {accuracy:.1%}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print("           " + "  ".join(f"{c:>8}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>10} " + "  ".join(f"{v:>8}" for v in confusion[i]))
    for cls, m in per_class.items():
        print(f"  {cls:>10}: precision={m['precision']:.2f} recall={m['recall']:.2f} "
              f"(n={m['support']})")

    torch.save(model.state_dict(), ARTIFACTS_DIR / "damage_classifier.pt")

    summary = {
        "n_train": len(train_idx), "n_test": len(test_idx), "n_epochs": N_EPOCHS,
        "test_accuracy": round(accuracy, 4),
        "confusion_matrix": confusion.tolist(), "classes": CLASSES,
        "per_class": per_class,
    }
    with open(REPORTS_DIR / "vision_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(REPORTS_DIR / "vision_evaluation.md", "w", encoding="utf-8") as f:
        f.write(f"# Damage Severity CNN - Evaluation\n\nTest accuracy: **{accuracy:.1%}** "
                f"(n_test={len(test_idx)})\n\n")
        f.write("| Class | Precision | Recall | Support |\n|---|---|---|---|\n")
        for cls, m in per_class.items():
            f.write(f"| {cls} | {m['precision']:.2f} | {m['recall']:.2f} | {m['support']} |\n")

    print(f"\nSaved model to {ARTIFACTS_DIR / 'damage_classifier.pt'} "
          f"and evaluation to {REPORTS_DIR / 'vision_evaluation.json'}")


if __name__ == "__main__":
    main()
