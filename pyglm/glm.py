'''base class for glms'''
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import fmin_bfgs

class GLMBase(BaseEstimator):
    @property
    def coef_(self):
        if self.fit_intercept:
            return self.coefficients[1:]
        return self.coefficients

    @coef_.setter
    def coef_(self, coef):
        if coef.shape != self.coefficients.shape:
            raise ValueError('coef shape does not match')
        self.coefficients = coef

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.coefficients[0]
        return None


def poisson_score(x, y, thetas):
    '''
    https://en.wikipedia.org/wiki/Poisson_regression
    simplified version of negative log likilihood, 
    ignore term that doesn't depend on model params (log(y!))
    '''
    thetas = thetas.reshape(-1, 1)
    score_i = np.exp(x @ thetas) - (y * (x @ thetas))
    return  np.mean(score_i)

def poisson_score_grad(x, y, thetas):
    '''
    partial derivatives of score with respect to thetas
    '''
    thetas = thetas.reshape(-1, 1)
    grad_i = (x * np.exp(x @ thetas)) - (y * x)
    return np.sum(grad_i, axis=0)


class PoissonGLM(GLMBase):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None


    def _add_intercept(self, X):
        n_rows = X.shape[0]
        intercept = np.ones((n_rows, 1))
        return np.hstack([intercept, X])

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.coefficients = fmin_bfgs(
            f=partial(poisson_score, X, y),
            x0=np.zeros(X.shape[1]),
            fprime=partial(poisson_score_grad, X, y))
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return np.exp(X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return poisson_score(X, y, self.coefficients)
            
