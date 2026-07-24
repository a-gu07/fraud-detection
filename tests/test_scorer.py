import pytest
import numpy as np
from app.scoring import MahalanobisScorer

@pytest.fixture
def lower_scorer():
    mean = np.array([10.0, 5.0])
    covariance = np.array([[4, 2], [2, 3]])  # cov inverse is [[0.375, −0.25], [−0.25, 0.5]]
    lower = np.linalg.cholesky(covariance)   # lower is [[2, 0], [1, √2]]
    return MahalanobisScorer(mean, lower)

@pytest.fixture
def global_scorer():
    mean = np.array([10.0, 5.0])
    lower = np.array([[2, 0], [1, np.sqrt(2)]])
    global_cov = np.array([[9, 0], [0, 9]])
    return MahalanobisScorer(mean, lower, global_cov)


class TestScoringOutputs:
    def test_mean_score(self, lower_scorer, global_scorer):
        lower = lower_scorer
        glob = global_scorer
        mean = np.array([10.0, 5.0])
        assert glob.score(mean) == 0
        assert lower.score(mean) == 0
    def test_mean_nonnegative(self, lower_scorer, global_scorer):
        lower = lower_scorer
        glob = global_scorer
        quadrant_means = [np.array([1, 1]), np.array([-1, 1]), np.array([-1, -1]), np.array([1, -1])]
        assert all(glob.score(mean) >= 0 for mean in quadrant_means)
        assert all(lower.score(mean) >= 0 for mean in quadrant_means)
    def test_score_output(self, lower_scorer, global_scorer):
        x = np.array([5, 10])
        glob = global_scorer
        lower = lower_scorer
        glob_result = glob.score(x)
        lower_result = lower.score(x)
        assert glob_result == pytest.approx(np.sqrt((x - glob.mean).T @ np.linalg.inv(glob.lower @ glob.upper) @ (x - glob.mean)))
        assert lower_result == pytest.approx(np.sqrt((x - lower.mean).T @ np.linalg.inv(lower.lower @ lower.upper) @ (x - lower.mean)))
    def test_score_symmetry(self, lower_scorer, global_scorer):
        glob = global_scorer
        lower = lower_scorer
        delta = np.array([0.5, 0.5])
        assert glob.score(glob.mean + delta) == pytest.approx(glob.score(glob.mean - delta))
        assert lower.score(lower.mean + delta) == pytest.approx(lower.score(lower.mean - delta))


class TestUpdateChanges:
    def test_instant_update(self, lower_scorer, global_scorer):
        glob = global_scorer
        lower = lower_scorer
        x = np.array([500, 500])
        del_t = 0
        glob_mean, glob_lower, glob_upper = glob.mean, glob.lower, glob.upper
        lower_mean, lower_lower, lower_upper = lower.mean, lower.lower, lower.upper
        glob.update(x, del_t)
        lower.update(x, del_t)
        assert np.array_equal(glob_mean, glob.mean) and np.array_equal(glob_lower, glob.lower) and np.array_equal(glob_upper, glob.upper)
        assert np.array_equal(lower_mean, lower.mean) and np.array_equal(lower_lower, lower.lower) and np.array_equal(lower_upper, lower.upper)
    def test_short_time_update(self, global_scorer):
        glob = global_scorer
        x = np.array([12, 6])
        del_t = glob.tau / 100 # weight w ~ 1%, dont expect much change in mean and covariance
        old_mean = glob.mean
        old_cov = glob.lower @ glob.upper
        glob.update(x, del_t)
        assert np.linalg.norm(old_mean - glob.mean) > 0 and np.linalg.norm(old_mean - glob.mean) < 0.1 * np.linalg.norm(old_mean - x)
        assert np.linalg.norm(old_cov - (glob.lower @ glob.upper)) > 0 and abs(np.linalg.norm(old_cov) - np.linalg.norm(glob.lower @ glob.upper)) < 0.1 * np.linalg.norm(old_cov)
        
    def test_long_time_update(self, global_scorer):
        glob = global_scorer
        x = np.array([50, 50])
        del_t = 100 * glob.tau # weight w close to 1
        old_mean = glob.mean
        glob.update(x, del_t)
        assert np.linalg.norm(old_mean - glob.mean) > 0 and np.allclose(x, glob.mean)
        assert np.allclose(glob.lower @ glob.upper, glob.global_cov)
    
    def test_global_cov_unchanged(self, global_scorer):
        glob = global_scorer
        x1, x2, x3 = np.array([50, 50]), np.array([-10, -10]), np.array([1000, -1000])
        t1, t2, t3 = 0, 500, 13
        cov1 = glob.global_cov
        glob.update(x1, t1)
        cov2 = glob.global_cov
        glob.update(x2, t2)
        cov3 = glob.global_cov
        glob.update(x3, t3)
        cov4 = glob.global_cov
        assert np.array_equal(cov1, cov2) and np.array_equal(cov1, cov3) and np.array_equal(cov1, cov4)


class TestInverseStability:
    def test_stability(self, global_scorer):
        x = np.array([50, 50])
        glob = global_scorer
        old_global = glob.global_cov
        del_t = 1e6 * glob.tau
        glob.update(x, del_t)
        assert np.allclose(old_global, glob.lower @ glob.upper)

class TestShape:
    def test_lower_shape(self, lower_scorer):
        lower = lower_scorer
        assert lower.mean.shape == (2,)
        assert lower.lower.shape == (2,2)
        assert lower.upper.shape == (2,2)
        assert lower.global_cov.shape == (2,2)
    def test_global_shape(self, global_scorer):
        glob = global_scorer
        assert glob.mean.shape == (2,)
        assert glob.lower.shape == (2,2)
        assert glob.upper.shape == (2,2)
        assert glob.global_cov.shape == (2,2)
    def test_transpose(self, lower_scorer, global_scorer):
        lower = lower_scorer
        glob = global_scorer
        assert np.array_equal(lower.lower.T, lower.upper)
        assert np.array_equal(glob.lower.T, glob.upper)
    def test_valid_cov(self, lower_scorer, global_scorer):
        """
        The covariance matrix must be symmetric by properties of covariance
        All eigenvalues of the covariance need to be positive, since the covariance needs to be inverted
        """
        lower = lower_scorer
        glob = global_scorer
        assert np.allclose(lower.lower @ lower.upper, (lower.lower @ lower.upper).T)
        assert np.allclose(glob.lower @ glob.upper, (glob.lower @ glob.upper).T)
        assert all(eig > 0 for eig in np.linalg.eigvalsh(lower.lower @ lower.upper))
        assert all(eig > 0 for eig in np.linalg.eigvalsh(glob.lower @ glob.upper))


class TestEWMAUpdate:
    def test_ewma_reference(self, lower_scorer):
        lower = lower_scorer
        # hand calculating updates
        x = np.array([50, 50])
        del_t = 360
        w = 1 - np.e ** ((-1 * del_t) / 3600)
        new_mean = (1-w) * lower.mean + w * x
        delta = x - lower.mean
        new_cov = (1 - w) * (lower.lower @ lower.upper) + w * (1 - w) * np.outer(delta, delta)
        used_cov = (1 - w) * new_cov + w * lower.global_cov

        lower.update(x, del_t)
        assert np.allclose(new_mean, lower.mean)
        assert np.allclose(used_cov, lower.lower @ lower.upper)
    
    def test_warmstart(self, global_scorer):
        glob = global_scorer
        new_scorer = MahalanobisScorer.fit_ewma(glob)
        assert np.array_equal(glob.mean, new_scorer.mean)
        assert np.array_equal(glob.lower, new_scorer.lower)
        assert np.array_equal(glob.global_cov, new_scorer.global_cov)
    
    def test_no_alias(self, global_scorer):
        glob = global_scorer
        new_scorer = MahalanobisScorer.fit_ewma(glob)
        x = np.array([50, 50])
        del_t = 100
        old_mean, old_lower,old_global = glob.mean, glob.lower, glob.global_cov
        new_scorer.update(x, del_t)
        assert np.array_equal(old_mean, glob.mean) and np.array_equal(old_lower, glob.lower) and np.array_equal(old_global, glob.global_cov)
    


