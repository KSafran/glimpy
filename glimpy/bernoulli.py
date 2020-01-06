""" bernoulli GLM """
from functools import partial
import numpy as np
from scipy.optimize import fmin_bfgs
from .glm import GLMBase
from .scoring import bernoulli_score
from .optimization import bernoulli_irls
from .link import anti_logit


class BernoulliGLM(GLMBase):
    """Bernoulli Generalized Linear Model

    Fits a bernoulli distributed GLM 

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
        """Fits a bernoulli glm using

        Parameters
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)
        y: np.ndarray response values, shape (n_obs, 1)
        """ 
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.coefficients = bernoulli_irls(X, y).reshape(-1)
        return self

    def predict(self, X):
        """Predicts Bernoulli Model

        Parameters 
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)

        Returns
        =======
        np.ndarray of the predictions, shape (n_obs, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return anti_logit(X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        """Scores bernoulli Model

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
        return bernoulli_score(X, y, self.coefficients)
