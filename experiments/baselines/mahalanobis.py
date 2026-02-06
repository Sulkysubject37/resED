import numpy as np

class MahalanobisDetector:
    def __init__(self):
        self.mu = None
        self.inv_cov = None
        
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        # Add regularization for stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)
        
    def score(self, X):
        # Mahalanobis distance squared: (x-mu)T S^-1 (x-mu)
        diff = X - self.mu
        # Vectorized computation: sum((diff @ inv_cov) * diff, axis=1)
        left = np.dot(diff, self.inv_cov)
        dist_sq = np.sum(left * diff, axis=1)
        return np.sqrt(dist_sq)
