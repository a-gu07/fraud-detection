# fraud-detection

Real-time fraud detection system that scores transactions for anomalousness using
hand-implemented statistical methods — no black-box library calls for the core
scoring logic. Built as a Summer 2027 SWE internship project (target: Capital One,
Roblox, FAANG-tier).

**Status:** In progress.

## Approach

**Dataset:** Kaggle "Credit Card Fraud Detection" (ULB) — ~285,000 transactions,
~0.17% fraud. Features are `Time`, `Amount`, and PCA-anonymized `V1`-`V28`. Split
into a historical set (baseline statistics) and a streaming set (held-out, replayed
as if live).

**Layer 1 — Global baseline (Mahalanobis distance).** Fit a mean vector and
covariance matrix on non-fraud historical transactions. Score each new transaction
by its Mahalanobis distance, D² = (x-μ)ᵀΣ⁻¹(x-μ), computed via Cholesky
factorization rather than direct matrix inversion. Higher distance = more
anomalous.

**Layer 2 — Time-adaptive baseline (EWMA).** This dataset is PCA-anonymized with
no real user/account IDs, so true per-entity baselines aren't possible. Rather
than fabricate synthetic entities, this layer instead adapts to time-based variance, 
which was empirically validated before building it: transaction volume, `Amount`, 
fraud rate, and several `V`-feature means/variances all vary meaningfully by hour. 
The scorer updates online via continuous-time exponential decay (`w = 1 - exp(-Δt/τ)`), 
and shrinks its covariance estimate toward the fixed global covariance after long 
gaps to avoid numerical instability.

**Layer 3 — Combining scores and evaluation.** The global and adaptive scores are
combined per transaction by taking their maximum: a transaction is treated as
anomalous if *either* baseline flags it, since the adaptive layer exists
specifically to catch drift-related anomalies the global baseline misses, and
vice versa. The combined score is evaluated by sweeping every possible decision
threshold against the true labels and computing the resulting false-positive
rate, recall, and precision at each cutoff. This gives ROC-AUC ≈ 0.959 and
PR-AUC ≈ 0.564 (cross-checked against scikit-learn's `roc_auc_score` and
`average_precision_score`, used here purely as an external validation
reference, not as part of the detector itself). A single operating threshold is
then chosen by fixing a tolerance for false positives. At a 0.5% false-positive-rate tolerance,
the detector achieves 81.8% recall and 22.1% precision. That precision figure is low but also 
reasonable. With fraud representing  ~0.17% of transactions, false positives are drawn from a 
pool over 500 times larger than the pool of actual fraud, so even a low false-positive *rate* 
translates into a large false-positive *count* relative to the number of true positives found. 

**Known limitation(s):** The dataset, being PCA-anonymized, does not fully reflect real-world 
transactions. Our methods have relied on the fact that the covariance matrix is invertible.
A real-world system would need some way to guarantee that the covariance matrix of their features is
invertible. Additionally, the fraud-likelihood decision based on the outputted scores relies on knowing 
which transactions are fraudulent and which aren't (this is baked into the update rule), which is also 
not something real-world data provides. The reported evaluation metrics were also computed, and the
detection threshold selected, using the same streaming set rather than a separate held-out split
reserved purely for threshold tuning, so the recall/precision figures above are likely somewhat
optimistic relative to how the detector would perform on genuinely unseen data.

## Repo layout

```
app/            core scoring logic (MahalanobisScorer)
notebooks/      exploratory analysis + validation notebooks
tests/          pytest suite for app/scoring.py
data/           dataset splits (gitignored)
```

## Running tests

```
source .venv/bin/activate
pytest
```

