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

Global and adaptive scores are currently both computed and exposed side by side
for every streaming transaction; deciding how to combine them into one final
fraud-likelihood decision is deferred.

**Known limitation(s):** The dataset, being PCA-anonymized, does not fully reflect real-world 
transactions. Our methods have relied on the fact that the covariance matrix is invertible.
A real-world system would need some way to guarantee that the covariance matrix of their features is
invertible. Additionally, the fraud-likelihood decision based on the outputted scores relies on knowing 
which transactions are fraudulent and which aren't (this is baked into the update rule), which is also 
not something real-world data provides. 

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

