# Document Extraction: Regex Baseline vs. LLM

Regex evaluated on 320 claims (full dataset). LLM evaluated on 80 claims (representative sample - see honesty notes).

| Field | Regex Accuracy | LLM Accuracy |
|---|---|---|
| policyholder | 100.0% | 98.8% |
| policy_number | 100.0% | 100.0% |
| category | 100.0% | 100.0% |
| item_description | 100.0% | 100.0% |
| item_value | 100.0% | 90.0% |
| claimed_amount | 100.0% | 96.2% |

**Overall: Regex 100.0% vs. LLM 97.5%**
