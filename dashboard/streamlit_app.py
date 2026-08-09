"""
streamlit_app.py
-------------------
The unified ops dashboard for the AutoClaim Intelligence Platform - four
tabs, all reading live from what the rest of the system actually
produced (dbt's governed fct_claims table via scripts/compute_metrics.py,
never a separate hand-written query):

  Claims Ops           - the flagged-claims queue an adjuster would work from
  Document AI            - regex baseline vs. LLM extraction accuracy
  Computer Vision          - the damage-severity CNN's confusion matrix
  Data Platform Health       - dbt contract/test status, the governed metrics

Run with:
    streamlit run dashboard/streamlit_app.py --server.port 8503
"""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.compute_metrics import compute_metrics  # noqa: E402

REPORTS_DIR = ROOT / "reports"
DB_PATH = ROOT / "database" / "warehouse.duckdb"

st.set_page_config(page_title="AutoClaim Intelligence Platform", layout="wide", page_icon="🏥")


def load_json(path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


st.title("🏥 AutoClaim Intelligence Platform")
st.caption("Document AI + Computer Vision + a governed dbt semantic layer, unified around "
           "insurance claims processing - reconciled by a cross-modal agent.")

tab_ops, tab_doc, tab_cv, tab_platform = st.tabs(
    ["📋 Claims Ops", "📄 Document AI", "🖼️ Computer Vision", "🗄️ Data Platform Health"])

# ============================================================================
# TAB 1: Claims Ops
# ============================================================================
with tab_ops:
    if not DB_PATH.exists():
        st.info("Run the pipeline first: `python scripts/run_pipeline.py`")
    else:
        metrics = compute_metrics()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Claims", int(metrics["claim_count"]))
        k2.metric("Total Claimed", f"${metrics['total_claimed_amount']:,.0f}")
        k3.metric("Flagged Claims", int(metrics["flagged_claim_count"]))
        k4.metric("Flagged Rate", f"{metrics['flagged_claim_rate']:.1%}")

        st.divider()
        st.subheader("🚩 Flagged Claims Queue")
        flags = load_json(REPORTS_DIR / "reconciliation_flags.json", default=[])
        if flags:
            flags_df = pd.DataFrame(flags)
            st.dataframe(
                flags_df[["claim_id", "predicted_severity", "item_value", "claimed_amount",
                          "claimed_fraction"]],
                use_container_width=True, height=250)
            selected = st.selectbox("View explanation for", flags_df.claim_id.tolist())
            explanation = flags_df[flags_df.claim_id == selected].explanation.iloc[0]
            st.info(explanation)
        else:
            st.caption("No flagged claims yet - run agents/claim_reconciliation_agent.py.")

        st.divider()
        st.subheader("Claims by Category & Predicted Severity")
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        claims_df = conn.execute("SELECT * FROM main.fct_claims").fetchdf()
        conn.close()
        fig = px.histogram(claims_df, x="category", color="predicted_severity", barmode="group",
                            category_orders={"predicted_severity": ["none", "minor", "moderate", "severe"]})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: Document AI
# ============================================================================
with tab_doc:
    extraction_eval = load_json(REPORTS_DIR / "extraction_evaluation.json")
    if extraction_eval is None:
        st.info("Run `python document_ai/evaluate_extraction.py` first.")
    else:
        st.subheader("Regex Baseline vs. LLM Extraction Accuracy")
        st.caption(f"Regex evaluated on {extraction_eval['n_regex_evaluated']} claims (full dataset). "
                   f"LLM evaluated on {extraction_eval['n_llm_evaluated']} claims "
                   f"(representative sample - see README honesty notes).")
        k1, k2 = st.columns(2)
        k1.metric("Regex Overall Accuracy", f"{extraction_eval['regex_overall_accuracy']:.1%}")
        k2.metric("LLM Overall Accuracy", f"{extraction_eval['llm_overall_accuracy']:.1%}")

        fields = list(extraction_eval["regex_accuracy_by_field"].keys())
        cmp_df = pd.DataFrame([
            {"field": f, "approach": "Regex", "accuracy": extraction_eval["regex_accuracy_by_field"][f]}
            for f in fields
        ] + [
            {"field": f, "approach": "LLM", "accuracy": extraction_eval["llm_accuracy_by_field"].get(f, 0)}
            for f in fields
        ])
        fig = px.bar(cmp_df, x="field", y="accuracy", color="approach", barmode="group",
                     color_discrete_map={"Regex": "#8896A6", "LLM": "#1F3A5F"})
        fig.update_layout(height=400, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: Computer Vision
# ============================================================================
with tab_cv:
    vision_eval = load_json(REPORTS_DIR / "vision_evaluation.json")
    if vision_eval is None:
        st.info("Run `python vision/train_damage_classifier.py` first.")
    else:
        st.subheader("Damage Severity CNN - Test Accuracy")
        st.metric("Test Accuracy", f"{vision_eval['test_accuracy']:.1%}",
                   f"n_test={vision_eval['n_test']}")

        classes = vision_eval["classes"]
        cm = pd.DataFrame(vision_eval["confusion_matrix"], index=classes, columns=classes)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True", color="Count"),
                         x=classes, y=classes, color_continuous_scale="Blues")
        fig.update_layout(height=420, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

        per_class_df = pd.DataFrame(vision_eval["per_class"]).T.reset_index().rename(columns={"index": "class"})
        st.dataframe(per_class_df, use_container_width=True)

# ============================================================================
# TAB 4: Data Platform Health
# ============================================================================
with tab_platform:
    st.subheader("Governed Semantic Layer Metrics")
    st.caption("Every number here comes from ONE definition in "
               "dbt_project/models/marts/metrics.yml, computed by scripts/compute_metrics.py - "
               "not a separate ad hoc query written for this dashboard.")
    if DB_PATH.exists():
        metrics = compute_metrics()
        st.json(metrics)
    else:
        st.info("Run the pipeline first.")

    st.divider()
    st.subheader("Data Contract")
    st.markdown("""
    `fct_claims` has an **enforced dbt model contract** (`config: {contract: {enforced: true}}`
    in `models/marts/_marts.yml`) - every column's data type is declared explicitly, and `dbt build`
    fails the build if the model's actual output doesn't match. This is what keeps a silent upstream
    schema change from becoming a silent downstream dashboard bug.
    """)

    st.subheader("Lineage")
    st.code("""
raw.claims_ground_truth ─┐ (eval only, never in marts)
raw.extraction_llm ───────┼─→ stg_extraction_llm ────┐
raw.extraction_regex ─────┼─→ stg_extraction_regex ──┼─→ int_claims_reconciled ─→ fct_claims
raw.vision_predictions ───┼─→ stg_vision_predictions ┤
raw.reconciliation_flags ─┴─→ stg_reconciliation_flags┘
    """, language="text")
    st.caption("Full column-level lineage: `dbt docs generate && dbt docs serve` "
               "(not run in this build environment, but the manifest it reads from is real).")
