'''Gamma Generalized Linear Models

Fits a gamma distributed GLM by holding the shape parameter constant
and allowing the scale parameter to vary linearly with predictors.
'''
from functools import partial
import numpy as np
from scipy.optimize import fmin_bfgs
from .glm import GLMBase
from .scoring import gamma_inverse_score #gamma_inverse_score_grad
from .optimization import gamma_irls


class GammaGLM(GLMBase):
    """Gamma Generalized Linear Model

    Fits a gamma distributed GLM by holding the shape parameter
    constant. Shape parameter is set using the closed-form
    estimator found here
    https://en.wikipedia.org/wiki/Gamma_distribution#Closed-form_estimator
    it can be overidden by an argument to the GammaGLM 
    constructor

    Parameters
    =========
    fit_intercept: bool, default=True 
        whether to add an intercept column to X

    link: string, default='inverse'
        link function to use, one of ['inverse']
    
    shape: float, default=None
        shape parameter to use for fitting. Default behavior uses
        closed form estimate for k on response data.

    method: string, default='irls'
        method for fitting gamma glm, 'irls' or 'bfgs'

    Attributes
    =========
    coef_: array of shape (n_features, )
        estimated coeffients of the gamma model, does not
        include the intercept coefficient

    intercept_: float
        estimated model intercept

    coefficients: array of shape (n_features + 1,)
        estimated coefficients including the intercept
    """ 

    def __init__(self, fit_intercept=True, link='inverse', shape=None, method='irls'):
        self.fit_intercept = fit_intercept
        self.link = link
        self.coefficients = None
        self.shape = shape
        self.method = method

    def fit(self, X, y):
        """Fits a gamma glm using bfgs

        Parameters
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)
        y: np.ndarray response values, shape (n_obs, 1)
        """ 
        if self.shape is None:
            self.shape = self.estimate_shape(y)
        if self.fit_intercept:
            X = self._add_intercept(X)
        if self.method == 'irls':
            self.coefficients = gamma_irls(X, y)
            return self

        self.coefficients = fmin_bfgs(
            f=partial(gamma_inverse_score, X, y, self.shape),
            x0=np.ones(X.shape[1]),
            # fprime=partial(gamm_score_grad, X, y),
        )
        return self

    def predict(self, X, return_scale=False):
        """Predicts Gamma Model

        Parameters 
        ==========
        X: np.ndarray of predictors, shape (n_obs, n_features)

        return_scale: bool, default=False
            return the scale estimate rather than expected value

        Returns
        =======
        np.ndarray of the predictions, shape (n_obs, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return 1.0/(X @ self.coefficients.reshape(-1, 1)) * self.shape

    def score(self, X, y):
        """Scores Gamma Model

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
        return gamma_inverse_score(X, y, self.shape, self.coefficients)

    def estimate_shape(self, y):
        '''Estimates shape parameter of y data

        closed form estimate of gamma shape for observed response
        https://en.wikipedia.org/wiki/Gamma_distribution#Closed-form_estimators

        Parameters
        ==========
        y: np.ndarray response values, shape (n_obs, 1)

        Returns
        =======
        gamma shape parameter estimate, float
        '''
        N = len(y)
        sum_y = y.sum()
        sum_ln_y = np.log(y).sum()
        sum_y_ln_y = (y * np.log(y)).sum()
        return (N * sum_y) / ((N * sum_y_ln_y) - (sum_y * sum_ln_y))


