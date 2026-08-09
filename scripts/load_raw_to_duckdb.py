"""
load_raw_to_duckdb.py
------------------------
Loads every raw output this platform's pipelines produce - the ground
truth, both document-extraction approaches, the CV predictions, and the
reconciliation flags - into raw DuckDB tables that dbt_project/ then
builds the governed staging -> intermediate -> marts layers on top of.
This is the same "land raw, then transform in dbt" separation used in
every prior dbt-based project in this portfolio.

Run this AFTER: scripts/generate_claims_dataset.py, document_ai/ocr_extract.py,
document_ai/regex_extraction.py, document_ai/llm_field_extraction.py,
vision/train_damage_classifier.py (+ predict), agents/claim_reconciliation_agent.py

Output: database/warehouse.duckdb (raw.* tables)
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DATABASE_DIR = ROOT / "database"
DATABASE_DIR.mkdir(exist_ok=True)

conn = duckdb.connect(str(DATABASE_DIR / "warehouse.duckdb"))
conn.execute("CREATE SCHEMA IF NOT EXISTS raw")

# ground truth
gt = pd.read_csv(DATA_DIR / "claims_ground_truth.csv")
conn.execute("CREATE OR REPLACE TABLE raw.claims_ground_truth AS SELECT * FROM gt")

# LLM extraction
with open(DATA_DIR / "llm_extracted_fields.json", encoding="utf-8") as f:
    llm_raw = json.load(f)
llm_rows = []
for claim_id, fields in llm_raw.items():
    row = {"claim_id": claim_id}
    row.update(fields)
    llm_rows.append(row)
llm_df = pd.DataFrame(llm_rows)
conn.execute("CREATE OR REPLACE TABLE raw.extraction_llm AS SELECT * FROM llm_df")

# Regex extraction
with open(DATA_DIR / "regex_extracted_fields.json", encoding="utf-8") as f:
    regex_raw = json.load(f)
regex_rows = []
for claim_id, fields in regex_raw.items():
    row = {"claim_id": claim_id}
    row.update(fields)
    regex_rows.append(row)
regex_df = pd.DataFrame(regex_rows)
conn.execute("CREATE OR REPLACE TABLE raw.extraction_regex AS SELECT * FROM regex_df")

# Vision predictions
with open(REPORTS_DIR / "vision_predictions.json", encoding="utf-8") as f:
    vision_raw = json.load(f)
vision_rows = [{"claim_id": k, **v} for k, v in vision_raw.items()]
vision_df = pd.DataFrame(vision_rows)
conn.execute("CREATE OR REPLACE TABLE raw.vision_predictions AS SELECT * FROM vision_df")

# Reconciliation flags
with open(REPORTS_DIR / "reconciliation_flags.json", encoding="utf-8") as f:
    flags_raw = json.load(f)
if flags_raw:
    flags_df = pd.DataFrame(flags_raw)
else:
    flags_df = pd.DataFrame(columns=["claim_id", "predicted_severity", "item_value",
                                       "claimed_amount", "claimed_fraction", "explanation"])
conn.execute("CREATE OR REPLACE TABLE raw.reconciliation_flags AS SELECT * FROM flags_df")

print("Loaded raw tables into database/warehouse.duckdb:")
for tbl in ["claims_ground_truth", "extraction_llm", "extraction_regex",
            "vision_predictions", "reconciliation_flags"]:
    n = conn.execute(f"SELECT COUNT(*) FROM raw.{tbl}").fetchone()[0]
    print(f"  raw.{tbl}: {n} rows")

conn.close()
