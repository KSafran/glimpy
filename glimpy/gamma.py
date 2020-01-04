'''
gamma distributed glms

to fit gamma distributed GLMs we hold the shape parameter constant
and allow the scale parameter to vary. In general, this can be used
with positive right-skewed data where the variance is proportional to
the square of the mean
'''
from functools import partial
import numpy as np
from scipy.optimize import fmin_bfgs
from .glm import GLMBase
from .scoring import gamma_inverse_score #gamma_inverse_score_grad


class GammaGLM(GLMBase):
    """
    a class for fitting poisson GLM based on the scikit-learn API
    """

    def __init__(self, fit_intercept=True, link='inverse', shape=None):
        """
        fit_intercept: Bool - whether to add an intercept column
        to the X ndarray when fitting
        """
        self.fit_intercept = fit_intercept
        self.link = link
        self.coefficients = None
        self.shape = shape

    def fit(self, X, y):
        """
        fits a gamma glm using bfgs

        X: two dimensional np.ndarray of predictors
        y: ndarray two dimensional np.ndarray response shape = (n, 1)
        """ 
        if self.shape is None:
            self.shape = self.estimate_shape(y)
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.coefficients = fmin_bfgs(
            f=partial(gamma_inverse_score, X, y, self.shape),
            x0=np.ones(X.shape[1]),
            # fprime=partial(gamm_score_grad, X, y),
        )
        return self

    def predict(self, X):
        """
        predicts conditional expected value of gamma scale 
        given X
        multiply by self.shape to get expected value
        
        X: two dimesional nd.array of predictors
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return 1.0/(X @ self.coefficients.reshape(-1, 1))

    def score(self, X, y):
        """
        scores a gamma glm model using variation of negative 
        log likelihood that ignores terms that dont depend on
        model parameters, and holds shape constant

        X: two dimensional np.ndarray of predictors
        y: ndarray two dimensional np.ndarray response shape = (n, 1)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return gamma_inverse_score(X, y, self.shape, self.coefficients)

    def estimate_shape(self, y):
        '''
        closed form estimate of gamma shape for observed response
        https://en.wikipedia.org/wiki/Gamma_distribution#Closed-form_estimators
        y: np.ndarray of postive observed response values
        '''
        N = len(y)
        sum_y = y.sum()
        sum_ln_y = np.log(y).sum()
        sum_y_ln_y = (y * np.log(y)).sum()
        return (N * sum_y) / ((N * sum_y_ln_y) - (sum_y * sum_ln_y))


