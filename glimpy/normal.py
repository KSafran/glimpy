'''Normal Generalized Linear Models

Equivalent to OLS Regression
'''
from functools import partial
import numpy as np
from .glm import GLMBase


class NormalGLM(GLMBase):
    """Normal Generalized Linear Model

    Fits a normal distributed GLM 

    Parameters
    =========
    fit_intercept: bool, default=True 
        whether to add an intercept column to X

    Attributes
    =========
    coef_: array of shape (n_features, )
        estimated coeffients of the model, does not
        include the intercept coefficient

    intercept_: float
        estimated model intercept

    coefficients: array of shape (n_features + 1,)
        estimated coefficients including the intercept
    """ 

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X, y):
        """Fits a normal glm using ols solution

        Parameters
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)
        y: np.ndarray response values, shape (n_obs, 1)
        """ 
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.coefficients = np.linalg.inv(X.T @ X) @ (X.T @ y).reshape(-1)
        return self

    def predict(self, X):
        """Predicts Normal Model

        Parameters 
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)

        Returns
        =======
        np.ndarray of the predictions, shape (n_obs, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return (X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        """Scores Normal Model Using Mean Squared Error

        Parameters
        ==========

        X: np.ndarray of predictors, shape (n_obs, n_features)
        y: np.ndarray response values, shape (n_obs, 1)

        Returns
        =======
        model score on X, y dataset, float
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        y_hat = self.predict(X)
        return np.mean((y - y_hat) ** 2)

