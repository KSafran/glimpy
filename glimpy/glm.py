"""base class for glms"""
import numpy as np
from sklearn.base import BaseEstimator


class GLMBase(BaseEstimator):
    '''
    Base class for GLM classes
    '''

    @property
    def coef_(self):
        '''
        An array of coefficient, excluding the intercept

        the way scikit-learn separates self.intercept_ from
        self.coef_ annoys me, but I'll match their API with
        these properties, and use self.coefficients for the 
        full nd.array of model cofficients
        '''
        if self.fit_intercept:
            return self.coefficients[1:]
        return self.coefficients

    @coef_.setter
    def coef_(self, coef):
        '''
        setting coefficients assumes you will also set
        the intercept
        '''
        if coef.shape != self.coefficients.shape:
            raise ValueError("coef shape does not match")
        self.coefficients = coef

    @property
    def intercept_(self):
        '''
        the models intercept if fit with one
        otherwise returns None
        '''
        if self.fit_intercept:
            return self.coefficients[0]
        return None

    @intercept_.setter
    def intercept_(self, intercept):
        '''
        set the intercept to a new value, only if 
        fit with intercept=True
        '''
        if not self.fit_intercept:
            raise AttributeError('not fit with intercept_')
        self.coefficients[0] = intercept
        
    def _add_intercept(self, X):
        """
        prepends the X ndarray with an column of 1s
        """
        n_rows = X.shape[0]
        intercept = np.ones((n_rows, 1))
        return np.hstack([intercept, X])



