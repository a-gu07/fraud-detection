from __future__ import annotations

import numpy as np
from scipy import linalg

class MahalanobisScorer:
    """
    Scores a transaction by its Mahalanobis distance from pre-determined stats
    (mean vector + covariance, represented via its Cholesky factor).
    """

    def __init__(self, mean: np.ndarray, lower: np.ndarray, global_cov: np.ndarray = None, tau: int = 3600 ):
        """
        mean: fitted mean vector, shape: (num_features,)
        lower: Cholesky factor L such that covariance == L @ L.T, shape: (num_features, num_features)
        """
        self.mean = mean
        self.lower = lower
        self.upper = lower.T
        # the covariance matrix Σ = E[(X - µ)(X - µ).T] = E[XX.T] - µµ.T, calculated on the historical data set
        self.global_cov = global_cov if global_cov is not None else lower @ lower.T 
        self.tau = tau

    def score(self, x: np.ndarray) -> float:
        """Return the Mahalanobis distance of transaction x from this baseline."""
        z = linalg.solve_triangular(self.lower, x - self.mean, lower=True)
        v = linalg.solve_triangular(self.upper, z, lower=False)
        return np.sqrt((x - self.mean).T @ v)

    # x is a new feature vector, and the elapsed time is the time from the previous feature to the current feature x
    def update(self, x: np.ndarray, elapsed_time: int):
        old_cov = self.lower @ self.upper
        # weight of the update takes the form of continuous decay with the formula w = 1 - exp(-∆t/τ)
        w = 1 - np.e ** ((-1 * elapsed_time) / self.tau)
        # Σ_new = (1-w)·Σ_old + w(1-w)·δδᵀ, where δ = = x - μ_old
        delta = x - self.mean
        new_cov = (1 - w) * old_cov + w * (1 - w) * np.outer(delta, delta)
        # new mean is µ_n = (1 - w)µ_o + wx 
        self.mean = (1-w) * self.mean + w * x
        # shrink toward the fixed, well-conditioned global covariance as w -> 1, since a long gap makes new_cov alone unreliable (risk of near-singularity)
        used_cov = (1 - w) * new_cov + w * self.global_cov
        self.lower = np.linalg.cholesky(used_cov)
        self.upper = self.lower.T
        

    @classmethod
    def fit_ewma(cls, global_scorer: MahalanobisScorer):
        """
        Alternate constructor: fit (or update) a time-varying baseline via
        online EWMA, then return a MahalanobisScorer built from the result.
        """

        return cls(mean=global_scorer.mean, lower=global_scorer.lower, global_cov=global_scorer.global_cov)
    
    @classmethod
    def fit(cls, data: np.ndarray):
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        lower = np.linalg.cholesky(cov)

        return cls(mean=mean, lower=lower, global_cov=cov)
