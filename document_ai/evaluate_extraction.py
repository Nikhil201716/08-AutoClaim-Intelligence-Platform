"""
evaluate_extraction.py
-------------------------
Compares both document-extraction approaches - the regex baseline
(document_ai/regex_extraction.py) and the LLM extraction
(document_ai/llm_field_extraction.py) - against the real ground truth
(data/claims_ground_truth.csv, generated alongside the PDFs and never
seen by either extractor). Text fields are scored by exact match;
numeric fields (item_value, claimed_amount) by a small tolerance (±$0.01)
to allow for legitimate float rounding, not sloppy scoring.

Only claims the LLM extraction actually ran on (see --sample-size in
llm_field_extraction.py) are included in the LLM comparison, so it's an
apples-to-apples comparison over the same claims, not the regex
baseline's full-dataset number against a partial LLM sample.

Output: reports/extraction_evaluation.json/.md
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEXT_FIELDS = ["policyholder", "policy_number", "category", "item_description"]
NUMERIC_FIELDS = ["item_value", "claimed_amount"]
ALL_FIELDS = TEXT_FIELDS + NUMERIC_FIELDS


def to_float(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def score(extracted: dict, truth: dict) -> dict:
    field_correct = {}
    for field in TEXT_FIELDS:
        field_correct[field] = str(extracted.get(field, "")).strip().lower() == \
            str(truth.get(field, "")).strip().lower()
    for field in NUMERIC_FIELDS:
        ev, tv = to_float(extracted.get(field)), truth.get(field)
        field_correct[field] = ev is not None and tv is not None and abs(ev - tv) < 0.01
    return field_correct


def main():
    truth_df = pd.read_csv(DATA_DIR / "claims_ground_truth.csv").set_index("claim_id")

    with open(DATA_DIR / "regex_extracted_fields.json", encoding="utf-8") as f:
        regex_extracted = json.load(f)
    with open(DATA_DIR / "llm_extracted_fields.json", encoding="utf-8") as f:
        llm_extracted = json.load(f)

    results = {"regex": {f: 0 for f in ALL_FIELDS}, "llm": {f: 0 for f in ALL_FIELDS}}
    n_regex, n_llm = 0, 0

    for claim_id, extracted in regex_extracted.items():
        truth = truth_df.loc[claim_id].to_dict()
        correct = score(extracted, truth)
        for field, is_correct in correct.items():
            results["regex"][field] += int(is_correct)
        n_regex += 1

    for claim_id, extracted in llm_extracted.items():
        if extracted.get("extraction_failed"):
            continue
        truth = truth_df.loc[claim_id].to_dict()
        correct = score(extracted, truth)
        for field, is_correct in correct.items():
            results["llm"][field] += int(is_correct)
        n_llm += 1

    summary = {
        "n_regex_evaluated": n_regex, "n_llm_evaluated": n_llm,
        "regex_accuracy_by_field": {f: round(results["regex"][f] / n_regex, 4) for f in ALL_FIELDS},
        "llm_accuracy_by_field": {f: round(results["llm"][f] / n_llm, 4) for f in ALL_FIELDS} if n_llm else {},
        "regex_overall_accuracy": round(sum(results["regex"].values()) / (n_regex * len(ALL_FIELDS)), 4),
        "llm_overall_accuracy": round(sum(results["llm"].values()) / (n_llm * len(ALL_FIELDS)), 4) if n_llm else None,
    }

    with open(REPORTS_DIR / "extraction_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "extraction_evaluation.md", "w", encoding="utf-8") as f:
        f.write("# Document Extraction: Regex Baseline vs. LLM\n\n")
        f.write(f"Regex evaluated on {n_regex} claims (full dataset). "
                f"LLM evaluated on {n_llm} claims (representative sample - see honesty notes).\n\n")
        f.write("| Field | Regex Accuracy | LLM Accuracy |\n|---|---|---|\n")
        for field in ALL_FIELDS:
            llm_acc = summary["llm_accuracy_by_field"].get(field)
            f.write(f"| {field} | {summary['regex_accuracy_by_field'][field]:.1%} | "
                    f"{f'{llm_acc:.1%}' if llm_acc is not None else 'n/a'} |\n")
        f.write(f"\n**Overall: Regex {summary['regex_overall_accuracy']:.1%} vs. "
                f"LLM {summary['llm_overall_accuracy']:.1%}**\n" if n_llm else "")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {REPORTS_DIR / 'extraction_evaluation.json'}")


if __name__ == "__main__":
    main()
