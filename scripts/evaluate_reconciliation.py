"""
evaluate_reconciliation.py
-----------------------------
Checks agents/claim_reconciliation_agent.py's flags against the real
ground truth (data/claims_ground_truth.csv's is_injected_mismatch column)
- the ONLY script in this platform allowed to look at that column, and
only for evaluation, never for the agent's own decision logic.

This measures something specific and important: the reconciliation
agent's flag rate (29 out of ~80 evaluated) is meaningfully higher than
the injected mismatch rate (18.8% of the full 320 claims) would predict.
This script exists to find out honestly why - is the agent catching real
injected mismatches (true positives) plus reasonable additional cases
from imperfect upstream extraction/CV, or is it mostly noise?

Output: reports/reconciliation_evaluation.json/.md
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

truth_df = pd.read_csv(DATA_DIR / "claims_ground_truth.csv").set_index("claim_id")

with open(DATA_DIR / "llm_extracted_fields.json", encoding="utf-8") as f:
    llm_extracted = json.load(f)
with open(REPORTS_DIR / "vision_predictions.json", encoding="utf-8") as f:
    vision_preds = json.load(f)
with open(REPORTS_DIR / "reconciliation_flags.json", encoding="utf-8") as f:
    flags = json.load(f)

flagged_ids = {f["claim_id"] for f in flags}

evaluated_ids = [
    cid for cid, fields in llm_extracted.items()
    if not fields.get("extraction_failed") and cid in vision_preds
]

tp = fp = fn = tn = 0
for claim_id in evaluated_ids:
    is_true_mismatch = bool(truth_df.loc[claim_id, "is_injected_mismatch"])
    is_flagged = claim_id in flagged_ids
    if is_true_mismatch and is_flagged:
        tp += 1
    elif not is_true_mismatch and is_flagged:
        fp += 1
    elif is_true_mismatch and not is_flagged:
        fn += 1
    else:
        tn += 1

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

# Why the false positives happen: CV misclassification and/or extraction
# error pushing a genuinely-consistent claim outside the agent's expected
# range - a real, explainable cause, not a mystery.
fp_causes = []
for claim_id in evaluated_ids:
    is_true_mismatch = bool(truth_df.loc[claim_id, "is_injected_mismatch"])
    is_flagged = claim_id in flagged_ids
    if is_flagged and not is_true_mismatch:
        true_severity = truth_df.loc[claim_id, "true_severity"]
        predicted_severity = vision_preds[claim_id]["predicted_severity"]
        fp_causes.append({
            "claim_id": claim_id,
            "cv_correct": true_severity == predicted_severity,
            "true_severity": true_severity,
            "predicted_severity": predicted_severity,
        })

n_fp_from_cv_error = sum(1 for c in fp_causes if not c["cv_correct"])

summary = {
    "n_evaluated": len(evaluated_ids),
    "confusion_matrix": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
    "precision": round(precision, 3),
    "recall": round(recall, 3),
    "f1_score": round(f1, 3),
    "false_positives_caused_by_cv_misclassification": n_fp_from_cv_error,
    "false_positives_total": fp,
    "false_positive_detail": fp_causes,
}

with open(REPORTS_DIR / "reconciliation_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(REPORTS_DIR / "reconciliation_evaluation.md", "w", encoding="utf-8") as f:
    f.write("# Reconciliation Agent Evaluation vs. Ground Truth\n\n")
    f.write(f"**Precision: {precision:.1%} · Recall: {recall:.1%} · F1: {f1:.3f}** "
            f"(n={len(evaluated_ids)} claims evaluated)\n\n")
    f.write(f"- True Positives (real mismatch, correctly flagged): {tp}\n")
    f.write(f"- False Positives (no real mismatch, incorrectly flagged): {fp}\n")
    f.write(f"  - of which {n_fp_from_cv_error} were caused by a CV misclassification "
            f"pushing an otherwise-consistent claim outside the expected range\n")
    f.write(f"- False Negatives (real mismatch, MISSED): {fn}\n")
    f.write(f"- True Negatives (no mismatch, correctly not flagged): {tn}\n")

print(json.dumps(summary, indent=2))
print(f"\nSaved to {REPORTS_DIR / 'reconciliation_evaluation.json'}")
