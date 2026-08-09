"""
generate_claims_dataset.py
-----------------------------
Generates the shared synthetic dataset both halves of this platform run
against: for each claim, a PDF claim form (for document_ai/ to extract
from) AND a paired damage photo (for vision/ to classify) - both derived
from the same ground-truth record, which is what makes cross-modal
reconciliation (agents/claim_reconciliation_agent.py) meaningful later:
it's checking whether two independently-built AI pipelines agree about
the same real underlying claim.

DELIBERATE INJECTED SCENARIO: for ~18% of claims, the claimed dollar
amount is set INCONSISTENT with the claim's true damage severity (e.g. a
high claim amount on a cosmetically-minor item, or a low claim amount on
a severely-damaged one) - a plausible mix of data-entry error and
potential-fraud signal. Neither the document extraction pipeline nor the
vision pipeline is told which claims these are; only
scripts/evaluate_reconciliation.py checks detection against this ground
truth, never the reconciliation agent's own logic.

Also deliberately messy: claim forms vary in layout/field order/phrasing
(not one clean template) and item categories/descriptions have realistic
wording variance - because a document-extraction pipeline that only works
on one exact template isn't demonstrating anything.

No real claimant or insurance data is used anywhere in this project.

Output:
  data/claims_ground_truth.csv
  data/claim_pdfs/CLAIM_*.pdf
  data/damage_images/{none,minor,moderate,severe}/CLAIM_*.png
"""

import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw

SEED = 11
rng = np.random.default_rng(SEED)
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "claim_pdfs"
IMG_DIR = DATA_DIR / "damage_images"
for d in [PDF_DIR] + [IMG_DIR / c for c in ["none", "minor", "moderate", "severe"]]:
    d.mkdir(parents=True, exist_ok=True)

N_CLAIMS = 320
MISMATCH_RATE = 0.18

CATEGORIES = {
    "Electronics": [("Laptop", 900), ("Smart TV", 650), ("Tablet", 350), ("Camera", 500)],
    "Appliance": [("Refrigerator", 1200), ("Washing Machine", 700), ("Microwave", 180)],
    "Furniture": [("Sofa", 800), ("Dining Table", 450), ("Office Chair", 220)],
    "Vehicle Part": [("Windshield", 400), ("Bumper", 550), ("Side Mirror", 150)],
}
FIRST_NAMES = ["James", "Maria", "Robert", "Linda", "David", "Sarah", "Michael", "Patricia",
               "John", "Jennifer", "Carlos", "Aisha", "Wei", "Priya", "Liam", "Emma"]
LAST_NAMES = ["Smith", "Garcia", "Johnson", "Williams", "Brown", "Davis", "Martinez", "Lee",
              "Wilson", "Anderson", "Thomas", "Patel", "Kim", "Nguyen", "Clark", "Rodriguez"]

SEVERITY_LEVELS = ["none", "minor", "moderate", "severe"]
SEVERITY_WEIGHTS = [0.10, 0.35, 0.35, 0.20]
# Roughly what a "reasonable" claim amount looks like per severity - a
# fraction of item value, with some natural variance.
SEVERITY_CLAIM_FRACTION = {"none": (0.0, 0.05), "minor": (0.08, 0.25),
                            "moderate": (0.30, 0.60), "severe": (0.65, 1.0)}

INCIDENT_PHRASES = [
    "Item was damaged during shipping.",
    "Damage occurred while in transit to the customer.",
    "Customer reports the item arrived in this condition.",
    "Damage noted upon delivery inspection.",
    "Item was dropped during handling at the warehouse.",
    "Water exposure during storage caused the damage.",
]

START_DATE = datetime(2026, 1, 1)


def make_claim_record(i):
    category = rng.choice(list(CATEGORIES.keys()))
    item_name, base_value = CATEGORIES[category][rng.integers(0, len(CATEGORIES[category]))]
    item_value = round(base_value * rng.uniform(0.85, 1.2), 2)

    severity = rng.choice(SEVERITY_LEVELS, p=SEVERITY_WEIGHTS)
    is_mismatch = rng.random() < MISMATCH_RATE

    if is_mismatch:
        # Deliberately claim inconsistent with true severity: either a
        # near-severe amount on a none/minor item, or a near-none amount
        # on a moderate/severe item.
        if severity in ("none", "minor"):
            lo, hi = SEVERITY_CLAIM_FRACTION["severe"]
        else:
            lo, hi = SEVERITY_CLAIM_FRACTION["none"]
    else:
        lo, hi = SEVERITY_CLAIM_FRACTION[severity]
    claimed_amount = round(item_value * rng.uniform(lo, max(hi, lo + 0.01)), 2)

    claim_date = (START_DATE + timedelta(days=int(rng.integers(0, 200)))).date()
    policyholder = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    policy_number = f"POL-{rng.integers(100000, 999999)}"
    claim_id = f"CLAIM_{10000 + i}"
    incident_desc = rng.choice(INCIDENT_PHRASES)

    return {
        "claim_id": claim_id, "policyholder": policyholder, "policy_number": policy_number,
        "claim_date": claim_date.isoformat(), "category": category, "item_description": item_name,
        "item_value": item_value, "true_severity": severity, "claimed_amount": claimed_amount,
        "incident_description": incident_desc, "is_injected_mismatch": bool(is_mismatch),
    }


def render_pdf(record, path):
    """A deliberately imperfect claim form - field order and phrasing vary
    across a few layout variants, the way real submitted forms would."""
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    layout_variant = rng.integers(0, 3)

    y = height - 80
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "INSURANCE CLAIM FORM")
    y -= 40
    c.setFont("Helvetica", 11)

    fields_common = [
        ("Claim ID", record["claim_id"]),
        ("Policyholder Name", record["policyholder"]),
        ("Policy Number", record["policy_number"]),
        ("Date of Claim", record["claim_date"]),
        ("Item Category", record["category"]),
        ("Item Description", record["item_description"]),
        (f"Item Value" if layout_variant != 2 else "Estimated Item Value",
         f"${record['item_value']:,.2f}"),
        ("Claimed Amount" if layout_variant == 0 else "Amount Claimed ($)",
         f"${record['claimed_amount']:,.2f}"),
    ]
    if layout_variant == 1:
        fields_common = fields_common[::-1]  # a genuinely different field order

    for label, value in fields_common:
        c.drawString(72, y, f"{label}:")
        c.drawString(260, y, str(value))
        y -= 22

    y -= 10
    c.drawString(72, y, "Incident Description:")
    y -= 18
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(90, y, record["incident_description"])
    c.save()


def render_damage_image(severity, path, size=96):
    """Parametric synthetic damage photo: a product silhouette with
    severity-correlated marks (scratches, dents, cracks) - not real
    photos, but a real, reproducible, controllable stand-in with a
    genuine visual difference between classes for a CNN to learn."""
    img = Image.new("RGB", (size, size), color=(235, 235, 230))
    draw = ImageDraw.Draw(img)
    # product body
    margin = size // 6
    draw.rectangle([margin, margin, size - margin, size - margin],
                    fill=(150, 150, 160), outline=(90, 90, 100))

    n_marks = {"none": 0, "minor": rng.integers(1, 3), "moderate": rng.integers(3, 6),
               "severe": rng.integers(6, 10)}[severity]
    mark_intensity = {"none": 0, "minor": 40, "moderate": 90, "severe": 160}[severity]

    for _ in range(n_marks):
        mark_type = rng.integers(0, 3)
        x0 = rng.integers(margin, size - margin)
        y0 = rng.integers(margin, size - margin)
        if mark_type == 0:  # scratch (line)
            x1 = x0 + rng.integers(-15, 15)
            y1 = y0 + rng.integers(-15, 15)
            draw.line([x0, y0, x1, y1], fill=(60, 60, 60), width=1)
        elif mark_type == 1:  # dent (dark ellipse)
            r = rng.integers(2, 5)
            shade = max(0, 90 - mark_intensity // 3)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=(shade, shade, shade))
        else:  # discoloration patch
            r = rng.integers(3, 7)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r],
                          fill=(120, 70 + mark_intensity // 4, 60))
    img.save(path)


records = [make_claim_record(i) for i in range(N_CLAIMS)]

for r in records:
    render_pdf(r, PDF_DIR / f"{r['claim_id']}.pdf")
    render_damage_image(r["true_severity"], IMG_DIR / r["true_severity"] / f"{r['claim_id']}.png")

import pandas as pd
df = pd.DataFrame(records)
df.to_csv(DATA_DIR / "claims_ground_truth.csv", index=False)

print(f"Generated {len(df)} claims -> {PDF_DIR} (PDFs) and {IMG_DIR} (images)")
print(f"Injected amount/severity mismatches: {df.is_injected_mismatch.sum()} "
      f"({df.is_injected_mismatch.mean():.1%})")
print(df.true_severity.value_counts())
