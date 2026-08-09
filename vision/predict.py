"""
predict.py
------------
Runs the trained damage-severity CNN (vision/artifacts/damage_classifier.pt)
over every claim's damage photo and saves a prediction per claim - this is
what agents/claim_reconciliation_agent.py and the dbt marts layer both
consume, so the CV pipeline's real output feeds the rest of the platform
rather than each part staying siloed.

Output: reports/vision_predictions.json  ({claim_id: {predicted_severity, confidence}})
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision.train_damage_classifier import DamageCNN, CLASSES, IMG_SIZE  # noqa: E402

IMG_DIR = ROOT / "data" / "damage_images"
ARTIFACTS_DIR = ROOT / "vision" / "artifacts"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_image_tensor(path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main():
    model = DamageCNN(n_classes=len(CLASSES))
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "damage_classifier.pt"))
    model.eval()

    predictions = {}
    image_paths = []
    for cls_dir in IMG_DIR.iterdir():
        if cls_dir.is_dir():
            image_paths.extend(cls_dir.glob("*.png"))

    with torch.no_grad():
        for path in sorted(image_paths):
            claim_id = path.stem
            tensor = load_image_tensor(path).unsqueeze(0)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax())
            predictions[claim_id] = {
                "predicted_severity": CLASSES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
            }

    with open(REPORTS_DIR / "vision_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Predicted severity for {len(predictions)} claims -> "
          f"{REPORTS_DIR / 'vision_predictions.json'}")


if __name__ == "__main__":
    main()
