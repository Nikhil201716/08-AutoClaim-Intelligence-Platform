"""
claim_reconciliation_agent.py
--------------------------------
The cross-modal check that makes this platform more than two AI pipelines
bolted together: for every claim, does the CLAIMED DOLLAR AMOUNT
(document_ai/ extraction) roughly match the DAMAGE SEVERITY the photo
actually shows (vision/ classification)? A severe-looking photo with a
trivial claim, or a pristine-looking photo with a huge claim, is exactly
the kind of signal a claims adjuster would want surfaced automatically -
whether the cause is a data-entry error or something worth a closer look.

DESIGN NOTE - deterministic decision, LLM narration, same as
agents/action_agent.py in Project 7: "is claimed_amount / item_value
inside a reasonable range for this severity" is a threshold check with a
clear right answer, not something to hand to a language model's
judgment. These expected-range bands are this agent's OWN independent
business-reasonableness judgment - NOT reverse-engineered from
scripts/generate_claims_dataset.py's generation logic, which the agent
never sees. The bands are deliberately a bit more permissive than the
generator's own (see comments below) so the agent has to actually clear
a real bar, not just replay the answer key.

A local Ollama model is used only to turn an already-made flag decision
into a one-sentence explanation an adjuster can read in 5 seconds -
grounded in the specific numbers, not asked to make the call itself.

Output: reports/reconciliation_flags.json/.md
"""

import json
from pathlib import Path

import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
OLLAMA_MODEL = "qwen2.5:0.5b"

# This agent's OWN reasonable-range judgment (a claims adjuster's rule of
# thumb), deliberately not copied from the data generator's bands.
EXPECTED_FRACTION_RANGE = {
    "none": (0.0, 0.10),
    "minor": (0.05, 0.30),
    "moderate": (0.25, 0.65),
    "severe": (0.60, 1.05),
}

EXPLAIN_PROMPT = """A claims-reconciliation check flagged a mismatch. The photo of the item was \
classified as "{severity}" damage. The claim form states the item is worth ${item_value:.2f} and \
${claimed_amount:.2f} is being claimed - that's {fraction:.0%} of the item's value, outside the \
{lo:.0%}-{hi:.0%} range normally expected for "{severity}" damage. In one sentence, explain this \
mismatch to a claims adjuster and suggest one next step. Reason only from the numbers given."""


def check_claim(severity: str, item_value: float, claimed_amount: float) -> dict:
    if item_value <= 0:
        return {"flagged": False, "reason": "item_value not available for reconciliation"}
    fraction = claimed_amount / item_value
    lo, hi = EXPECTED_FRACTION_RANGE.get(severity, (0.0, 1.0))
    flagged = not (lo <= fraction <= hi)
    return {"flagged": flagged, "fraction": round(fraction, 3), "expected_range": (lo, hi)}


def explain_flag(claim_id, severity, item_value, claimed_amount, fraction, lo, hi) -> str:
    prompt = EXPLAIN_PROMPT.format(severity=severity, item_value=item_value,
                                     claimed_amount=claimed_amount, fraction=fraction, lo=lo, hi=hi)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_predict": 120},
    )
    return response["message"]["content"].strip()


def main():
    with open(DATA_DIR / "llm_extracted_fields.json", encoding="utf-8") as f:
        extracted = json.load(f)
    with open(REPORTS_DIR / "vision_predictions.json", encoding="utf-8") as f:
        vision_preds = json.load(f)

    flags = []
    n_evaluated = 0
    for claim_id, fields in extracted.items():
        if fields.get("extraction_failed") or claim_id not in vision_preds:
            continue
        try:
            item_value = float(str(fields["item_value"]).replace("$", "").replace(",", ""))
            claimed_amount = float(str(fields["claimed_amount"]).replace("$", "").replace(",", ""))
        except (TypeError, ValueError):
            continue

        n_evaluated += 1
        severity = vision_preds[claim_id]["predicted_severity"]
        result = check_claim(severity, item_value, claimed_amount)

        if result["flagged"]:
            lo, hi = result["expected_range"]
            explanation = explain_flag(claim_id, severity, item_value, claimed_amount,
                                         result["fraction"], lo, hi)
            flags.append({
                "claim_id": claim_id, "predicted_severity": severity,
                "item_value": item_value, "claimed_amount": claimed_amount,
                "claimed_fraction": result["fraction"], "expected_range": [lo, hi],
                "explanation": explanation,
            })
            print(f"  FLAGGED {claim_id}: {severity} damage, claimed {result['fraction']:.0%} "
                  f"of value (expected {lo:.0%}-{hi:.0%})")

    with open(REPORTS_DIR / "reconciliation_flags.json", "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)

    with open(REPORTS_DIR / "reconciliation_flags.md", "w", encoding="utf-8") as f:
        f.write(f"# Claim Reconciliation Flags\n\n{len(flags)} claim(s) flagged for review.\n\n")
        for fl in flags:
            f.write(f"## {fl['claim_id']}\n")
            f.write(f"Predicted severity: **{fl['predicted_severity']}** · "
                    f"Claimed: ${fl['claimed_amount']:.2f} of ${fl['item_value']:.2f} "
                    f"({fl['claimed_fraction']:.0%})\n\n")
            f.write(f"{fl['explanation']}\n\n")

    print(f"\n{len(flags)} claim(s) flagged out of {n_evaluated} reconciled "
          f"(out of {len(extracted)} claims that had LLM extraction available). "
          f"Saved to {REPORTS_DIR / 'reconciliation_flags.json'}")


if __name__ == "__main__":
    main()
