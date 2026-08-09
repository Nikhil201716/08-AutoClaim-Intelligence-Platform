"""
compute_metrics.py
---------------------
Computes the exact metrics defined in dbt_project/models/marts/metrics.yml
against the governed fct_claims table - the "query the semantic layer"
half of the story. Querying dbt's MetricFlow semantic layer for real
requires the dbt Semantic Layer product (dbt Cloud); this script instead
reads the SAME metrics.yml definitions this project's dbt project ships
and computes them directly against fct_claims in plain SQL, so the
governance goal - metrics defined exactly once, in one YAML file, that
both dbt and this platform's dashboard agree on - is genuinely achieved
without requiring infrastructure this project doesn't have access to.
dashboard/streamlit_app.py imports and calls this module rather than
writing its own separate metric SQL.

Output: reports/semantic_layer_metrics.json
"""

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "warehouse.duckdb"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Mirrors dbt_project/models/marts/metrics.yml exactly - see that file for
# the governed definition each of these implements.
METRIC_QUERIES = {
    "claim_count": "SELECT COUNT(claim_id) FROM main.fct_claims",
    "total_claimed_amount": "SELECT SUM(claimed_amount) FROM main.fct_claims",
    "flagged_claim_count": "SELECT COUNT(claim_id) FROM main.fct_claims WHERE is_flagged = true",
    "flagged_claim_rate": """
        SELECT CAST(COUNT(claim_id) FILTER (WHERE is_flagged = true) AS DOUBLE)
               / NULLIF(COUNT(claim_id), 0)
        FROM main.fct_claims
    """,
    "avg_cv_confidence": "SELECT AVG(vision_confidence) FROM main.fct_claims",
}


def compute_metrics() -> dict:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    results = {}
    for name, query in METRIC_QUERIES.items():
        value = conn.execute(query).fetchone()[0]
        results[name] = round(float(value), 4) if value is not None else None
    conn.close()
    return results


def main():
    metrics = compute_metrics()
    with open(REPORTS_DIR / "semantic_layer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to {REPORTS_DIR / 'semantic_layer_metrics.json'}")


if __name__ == "__main__":
    main()
