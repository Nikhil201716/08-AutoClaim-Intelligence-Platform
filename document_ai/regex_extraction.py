"""
regex_extraction.py
----------------------
The baseline every "AI-powered" document extraction claim has to beat:
plain regex over the OCR'd text. Since claim forms are still fundamentally
"Label: Value" pairs (just with label wording/order variance across the 3
layout templates - see scripts/generate_claims_dataset.py), a
hand-maintained set of label aliases can plausibly do quite well here -
and it's worth honestly finding out whether it does, rather than assuming
an LLM is automatically the better tool (the same "know which tool
actually fits" judgment applied throughout this portfolio).

Output: data/regex_extracted_fields.json
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_TEXT_PATH = ROOT / "data" / "ocr_raw_text.json"
OUT_PATH = ROOT / "data" / "regex_extracted_fields.json"

# Each field maps to the label variants actually used across the 3 layout
# templates in scripts/generate_claims_dataset.py.
FIELD_PATTERNS = {
    "claim_id": [r"Claim ID:\s*(\S+)"],
    "policyholder": [r"Policyholder Name:\s*(.+)"],
    "policy_number": [r"Policy Number:\s*(\S+)"],
    "claim_date": [r"Date of Claim:\s*(\S+)"],
    "category": [r"Item Category:\s*(.+)"],
    "item_description": [r"Item Description:\s*(.+)"],
    "item_value": [r"(?:Item Value|Estimated Item Value):\s*\$?([\d,]+\.\d{2})"],
    "claimed_amount": [r"(?:Claimed Amount|Amount Claimed \(\$\)):\s*\$?([\d,]+\.\d{2})"],
}


def extract_fields(text: str) -> dict:
    result = {}
    for field, patterns in FIELD_PATTERNS.items():
        value = None
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                value = m.group(1).strip()
                break
        result[field] = value
    return result


def main():
    with open(RAW_TEXT_PATH, encoding="utf-8") as f:
        raw_texts = json.load(f)

    extracted = {claim_id: extract_fields(text) for claim_id, text in raw_texts.items()}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)

    n_complete = sum(1 for v in extracted.values() if all(v.values()))
    print(f"Regex-extracted fields for {len(extracted)} claims -> {OUT_PATH}")
    print(f"Claims with all fields found: {n_complete}/{len(extracted)}")


if __name__ == "__main__":
    main()
