""" Poisson GLM """
from functools import partial
import numpy as np
from scipy.optimize import fmin_bfgs
from .glm import GLMBase
from .scoring import poisson_score, poisson_score_grad


class PoissonGLM(GLMBase):
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

    def _add_intercept(self, X):
        """
        prepends the X ndarray with an column of 1s
        """
        n_rows = X.shape[0]
        intercept = np.ones((n_rows, 1))
        return np.hstack([intercept, X])

    def fit(self, X, y):
        """
        fits a poisson glm using bfgs

        X: two dimensional np.ndarray of predictors
        y: ndarray two dimensional np.ndarray response shape = (n, 1)
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
        """
        predicts conditional expected value of poisson distribution
        given X
        
        X: two dimesional nd.array of predictors
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return np.exp(X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        """
        scores a poisson glm model using variation of negative 
        log likelihood that ignores terms that dont depend on
        model parameters

        X: two dimensional np.ndarray of predictors
        y: ndarray two dimensional np.ndarray response shape = (n, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return poisson_score(X, y, self.coefficients)
