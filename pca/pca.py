"""
PCA implementation for learning the internals
"""
import numpy as np

class PCA:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        
        # 1- Subtract the mean from every column (mean centering)
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # 2- calculates convariance (needs samples as columns)
        covariance = np.cov(X.T)
        
        # 3- calculates eigenvalues and eigenvectors from convariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        
        # 4- transpose for easier calculations
        eigenvectors = eigenvectors.T
        
        # 5- sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # 6- Pick top n eigenvectors as principal components
        self.components = eigenvectors[0: self.n_components]
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
