"""
run_pipeline.py
------------------
One-command orchestrator for the full AutoClaim Intelligence Platform:
data generation -> document extraction (regex + LLM) -> CV training ->
cross-modal reconciliation -> dbt semantic layer -> preview images.

Usage:
    python scripts/run_pipeline.py [--llm-sample-size N]

Notes on timing: the LLM extraction and reconciliation-explanation steps
both call a local Ollama model per-claim, which is genuinely slow on
constrained hardware (observed: several seconds to over a minute per call
depending on system load - see README honesty notes). --llm-sample-size
controls how many of the 320 claims run through the LLM path; the regex
baseline and CV classifier always run on the full dataset.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = ROOT / "dbt_project"
PY = sys.executable
# Resolve dbt.exe relative to sys.executable rather than trusting PATH -
# the same defensive pattern used in every dbt-based project in this
# portfolio, since multiple Python installs on PATH can shadow each other.
_scripts_dir = Path(sys.executable).parent / "Scripts"
DBT_EXE = str(_scripts_dir / "dbt.exe") if (_scripts_dir / "dbt.exe").exists() else "dbt"


def run(label, args, cwd=ROOT):
    print(f"\n{'='*70}\n{label}...\n{'='*70}")
    result = subprocess.run([PY] + args, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\nStep failed (exit {result.returncode}): {' '.join(args)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-sample-size", type=int, default=80)
    args = parser.parse_args()

    run("1/9 Generating synthetic claims dataset (PDFs + damage images)",
        ["scripts/generate_claims_dataset.py"])
    run("2/9 Extracting raw text from claim PDFs", ["document_ai/ocr_extract.py"])
    run("3/9 Regex baseline extraction (full dataset)", ["document_ai/regex_extraction.py"])
    run("4/9 LLM extraction (Ollama, sampled)",
        ["document_ai/llm_field_extraction.py", "--sample-size", str(args.llm_sample_size)])
    run("5/9 Evaluating extraction accuracy vs. ground truth", ["document_ai/evaluate_extraction.py"])
    run("6/9 Training damage-severity CNN", ["vision/train_damage_classifier.py"])
    run("6b/9 Predicting damage severity for all claims", ["vision/predict.py"])
    run("7/9 Cross-modal claim reconciliation (Ollama)", ["agents/claim_reconciliation_agent.py"])
    run("8/9 Loading raw outputs into DuckDB", ["scripts/load_raw_to_duckdb.py"])

    print(f"\n{'='*70}\n9/9 Building dbt semantic layer (staging -> intermediate -> marts)...\n{'='*70}")
    dbt_result = subprocess.run([DBT_EXE, "build"], cwd=str(DBT_DIR))
    if dbt_result.returncode != 0:
        print("\ndbt build failed. Fix the issue above and rerun.")
        sys.exit(1)

    run("Computing governed semantic layer metrics", ["scripts/compute_metrics.py"])
    run("Generating preview images", ["scripts/generate_preview_images.py"])

    print("\n" + "=" * 70)
    print("Pipeline complete. Launch the dashboard to explore the results:")
    print("    streamlit run dashboard/streamlit_app.py --server.port 8503")
    print("=" * 70)


if __name__ == "__main__":
    main()
