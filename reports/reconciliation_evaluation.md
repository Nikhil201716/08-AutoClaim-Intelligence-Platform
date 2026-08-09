# Reconciliation Agent Evaluation vs. Ground Truth

**Precision: 65.5% · Recall: 100.0% · F1: 0.792** (n=80 claims evaluated)

- True Positives (real mismatch, correctly flagged): 19
- False Positives (no real mismatch, incorrectly flagged): 10
  - of which 2 were caused by a CV misclassification pushing an otherwise-consistent claim outside the expected range
- False Negatives (real mismatch, MISSED): 0
- True Negatives (no mismatch, correctly not flagged): 51
