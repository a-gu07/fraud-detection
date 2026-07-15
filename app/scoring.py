import numpy as np
from scipy import linalg

class MahalanobisScorer:
    """
    Scores a transaction by its Mahalanobis distance from pre-determined stats
    (mean vector + covariance, represented via its Cholesky factor).
    """

    def __init__(self, mean: np.ndarray, lower: np.ndarray):
        """
        mean: fitted mean vector, shape: (num_features,)
        lower: Cholesky factor L such that covariance == L @ L.T, shape: (num_features, num_features)
        """
        self.mean = mean
        self.lower = lower
        self.upper = lower.T

    def score(self, x: np.ndarray) -> float:
        """Return the Mahalanobis distance of transaction x from this baseline."""
        z = linalg.solve_triangular(self.lower, x - self.mean, lower=True)
        v = linalg.solve_triangular(self.upper, z, lower=False)
        return np.sqrt((x - self.mean).T @ v)

    @classmethod
    def fit_ewma(cls):
        """
        Alternate constructor: fit (or update) a per-entity baseline via
        online EWMA, then return a MahalanobisScorer built from the result.
        Required work: signature/params still to be written
        """
        # TODO
        raise NotImplementedError
