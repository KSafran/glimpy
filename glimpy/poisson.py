""" Poisson GLM """
from functools import partial
import numpy as np
from scipy.optimize import fmin_bfgs
from .glm import GLMBase
from .scoring import poisson_score, poisson_score_grad


class PoissonGLM(GLMBase):
    """Poisson Generalized Linear Model

    Fits a poisson distributed GLM 

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
        """Fits a poisson glm using bfgs

        Parameters
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)
        y: np.ndarray response values, shape (n_obs, 1)
        """ 
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.coefficients = fmin_bfgs(
            f=partial(poisson_score, X, y),
            x0=np.zeros(X.shape[1]),
            fprime=partial(poisson_score_grad, X, y),
        )
        return self

    def predict(self, X):
        """Predicts Poisson Model

        Parameters 
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)

        Returns
        =======
        np.ndarray of the predictions, shape (n_obs, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return np.exp(X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        """Scores Poisson Model

        Note: this score is a variation of negative log-likelihood that
        ignores terms that dont depent on model parameters.

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
        return poisson_score(X, y, self.coefficients)
