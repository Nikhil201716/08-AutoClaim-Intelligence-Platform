"""
llm_field_extraction.py
--------------------------
Structured field extraction via a local Ollama model, asked to return
JSON directly - the "AI-powered" half of the document intelligence
comparison against document_ai/regex_extraction.py's hand-maintained
label-alias baseline.

GUARDRAIL: the model's response is only accepted if it parses as valid
JSON containing all 8 expected keys. If it doesn't (malformed JSON,
missing keys, extra prose wrapped around the JSON), this module does NOT
guess or partially accept it - the claim is marked `extraction_failed`
for human review, the same never-silently-accept-a-bad-parse principle
used for every guardrail elsewhere in this portfolio.

Output: data/llm_extracted_fields.json
"""

import argparse
import json
import re
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent.parent
RAW_TEXT_PATH = ROOT / "data" / "ocr_raw_text.json"
OUT_PATH = ROOT / "data" / "llm_extracted_fields.json"
OLLAMA_MODEL = "qwen2.5:0.5b"

EXPECTED_KEYS = ["claim_id", "policyholder", "policy_number", "claim_date",
                  "category", "item_description", "item_value", "claimed_amount"]

PROMPT_TEMPLATE = """Extract fields from this insurance claim form text into JSON.

Claim form text:
---
{text}
---

Respond with ONLY a JSON object, no other text. The JSON MUST use exactly these 8 key names, \
spelled exactly like this, even if the form uses different wording for the label - map the form's \
label to the correct key below, do not invent your own key names:

{{
  "claim_id": "...",
  "policyholder": "...",
  "policy_number": "...",
  "claim_date": "...",
  "category": "...",
  "item_description": "...",
  "item_value": <number, no $ or commas>,
  "claimed_amount": <number, no $ or commas>
}}

("Item Value" or "Estimated Item Value" in the form both map to the "item_value" key. "Claimed \
Amount" or "Amount Claimed ($)" in the form both map to the "claimed_amount" key.)

JSON:"""


def _try_parse_json(raw: str) -> dict | None:
    # The model sometimes wraps JSON in ```json fences or adds a stray
    # sentence - pull out the first {...} block rather than requiring a
    # perfectly clean response.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_fields_llm(text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(keys=", ".join(EXPECTED_KEYS), text=text)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 250},
    )
    raw = response["message"]["content"].strip()
    parsed = _try_parse_json(raw)

    if parsed is None or not all(k in parsed for k in EXPECTED_KEYS):
        return {"extraction_failed": True, "raw_response": raw}

    parsed["extraction_failed"] = False
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None,
                         help="If set, only run extraction on this many claims (first N, "
                              "deterministic given the fixed generation seed) - the local "
                              "model is slow enough that a representative sample is more "
                              "practical than all 320 for iterative development.")
    args = parser.parse_args()

    with open(RAW_TEXT_PATH, encoding="utf-8") as f:
        raw_texts = json.load(f)

    if args.sample_size:
        raw_texts = dict(list(raw_texts.items())[:args.sample_size])

    extracted = {}
    n_failed = 0
    for i, (claim_id, text) in enumerate(raw_texts.items(), 1):
        result = extract_fields_llm(text)
        extracted[claim_id] = result
        if result.get("extraction_failed"):
            n_failed += 1
        if i % 20 == 0:
            print(f"  ...{i}/{len(raw_texts)} claims processed")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)

    print(f"\nLLM-extracted fields for {len(extracted)} claims -> {OUT_PATH}")
    print(f"Extraction failures (guardrail triggered): {n_failed}/{len(extracted)}")


if __name__ == "__main__":
    main()
