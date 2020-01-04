'''
normal distributed glms

same as scikit LinearRegression
Fit using closed form least squares solution
'''
from functools import partial
import numpy as np
from .glm import GLMBase


class NormalGLM(GLMBase):
    """
    a class for fitting poisson GLM based on the scikit-learn API
    """

    def __init__(self, fit_intercept=True):
        """
        fit_intercept: Bool - whether to add an intercept column
        to the X ndarray when fitting
        """
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X, y):
        """
        fits a normal glm using OLS

        X: two dimensional np.ndarray of predictors
        y: ndarray two dimensional np.ndarray response shape = (n, 1)
        """ 
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.coefficients = np.linalg.inv(X.T @ X) @ (X.T @ y).reshape(-1)
        return self

    def predict(self, X):
        """
        predicts mean of response given X
        
        X: two dimesional nd.array of predictors
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return (X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        """
        mean squared error

        X: two dimensional np.ndarray of predictors
        y: ndarray two dimensional np.ndarray response shape = (n, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        y_hat = self.predict(X)
        return np.mean((y - y_hat) ** 2)

