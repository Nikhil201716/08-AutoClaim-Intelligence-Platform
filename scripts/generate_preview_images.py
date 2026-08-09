"""
generate_preview_images.py
-----------------------------
Renders static PNG chart previews (matplotlib) straight from this
pipeline's real output for the README - this build environment has no
display to screenshot the live Streamlit app.

Output: ../screenshots/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
OUT_DIR = ROOT / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

NAVY, ACCENT, RED, GOLD, GREY = "#1F3A5F", "#2E6F40", "#C0392B", "#E1A100", "#8896A6"

with open(REPORTS_DIR / "extraction_evaluation.json", encoding="utf-8") as f:
    extraction_eval = json.load(f)
with open(REPORTS_DIR / "vision_evaluation.json", encoding="utf-8") as f:
    vision_eval = json.load(f)
with open(REPORTS_DIR / "semantic_layer_metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

# ------------------------------------------------------------------
# 1. KPI summary
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(16, 2.2))
cards = [
    ("Total Claims", f"{int(metrics['claim_count'])}"),
    ("Flagged Rate", f"{metrics['flagged_claim_rate']:.1%}"),
    ("LLM Extraction Acc.", f"{extraction_eval['llm_overall_accuracy']:.1%}"),
    ("CV Test Accuracy", f"{vision_eval['test_accuracy']:.1%}"),
    ("Total Claimed", f"${metrics['total_claimed_amount']:,.0f}"),
]
for ax, (label, value) in zip(axes, cards):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY, transform=ax.transAxes, zorder=0))
    ax.text(0.5, 0.68, label, ha="center", va="center", color="white", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.32, value, ha="center", va="center", color="white", fontsize=14, fontweight="bold", transform=ax.transAxes)
fig.suptitle("AutoClaim Intelligence Platform - Key Metrics", fontsize=12, color=NAVY, y=1.08)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_kpi_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 2. Extraction accuracy: regex vs LLM
# ------------------------------------------------------------------
fields = list(extraction_eval["regex_accuracy_by_field"].keys())
regex_vals = [extraction_eval["regex_accuracy_by_field"][f] for f in fields]
llm_vals = [extraction_eval["llm_accuracy_by_field"].get(f, 0) for f in fields]

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(fields))
width = 0.35
ax.bar(x - width/2, regex_vals, width, label="Regex Baseline", color=GREY)
ax.bar(x + width/2, llm_vals, width, label="LLM Extraction", color=NAVY)
ax.set_xticks(x)
ax.set_xticklabels(fields, rotation=20, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.set_title("Document Extraction Accuracy: Regex Baseline vs. LLM (per field)", color=NAVY, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "02_extraction_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 3. CV confusion matrix
# ------------------------------------------------------------------
classes = vision_eval["classes"]
cm = np.array(vision_eval["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Damage Severity CNN Confusion Matrix\n(Test Accuracy: {vision_eval['test_accuracy']:.1%})",
             color=NAVY, fontweight="bold")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_cv_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved 3 preview images to", OUT_DIR)
