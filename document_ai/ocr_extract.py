"""
ocr_extract.py
-----------------
Extracts raw text from every claim PDF - the "OCR" step of this pipeline.

Honesty note on the word "OCR": these PDFs have embedded, selectable text
(they were generated with reportlab, not scanned from paper), so
`pdfplumber` reads that text directly rather than doing true image-based
optical character recognition. The DOWNSTREAM problem this project
demonstrates - turning inconsistent, semi-structured raw text into
validated structured fields (document_ai/llm_field_extraction.py) - is
the same problem either way; true image OCR (Tesseract or similar) would
slot in as a drop-in replacement for this module's `extract_text()`
function without changing anything downstream. Being upfront about this
distinction matters more than pretending otherwise.

Output: data/ocr_raw_text.json  ({claim_id: raw_text})
"""

import json
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "claim_pdfs"
OUT_PATH = ROOT / "data" / "ocr_raw_text.json"


def extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def main():
    raw_texts = {}
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        claim_id = pdf_path.stem
        raw_texts[claim_id] = extract_text(pdf_path)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_texts, f, indent=2)

    print(f"Extracted text from {len(raw_texts)} PDFs -> {OUT_PATH}")
    sample_id = next(iter(raw_texts))
    print(f"\nSample ({sample_id}):\n{raw_texts[sample_id]}")


if __name__ == "__main__":
    main()
