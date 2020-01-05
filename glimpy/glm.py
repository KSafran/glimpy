"""Base GLM Class"""
import numpy as np
from sklearn.base import BaseEstimator


class GLMBase(BaseEstimator):
    '''Base GLM Class
    
    Extends the scikit-learn BaseEstimator class which allows
    for simple interaction with scikit-learn API. Properties 
    for coef_ and intercept_ attributes allow the classes to 
    conform to the scikit-learn API while allowing the 
    implementation to store all model coefficients together
    in a single `coefficients` attribute.
    '''

    @property
    def coef_(self):
        '''An array of coefficients, excludes the intercept.'''
        if self.fit_intercept:
            return self.coefficients[1:]
        return self.coefficients

    @coef_.setter
    def coef_(self, coef):
        '''Sets Model Coefficients

        This assumes you will also set the intercept
        '''
        if coef.shape != self.coefficients.shape:
            raise ValueError("coef shape does not match")
        self.coefficients = coef

    @property
    def intercept_(self):
        '''Model Intercept Parameter

        Assumes model fit with intercept, otherwise
        returns None
        '''
        if self.fit_intercept:
            return self.coefficients[0]
        return None

    @intercept_.setter
    def intercept_(self, intercept):
        '''Sets Model Intercept

        Only allowed if fit with fit_intercept=True
        '''
        if not self.fit_intercept:
            raise AttributeError('not fit with intercept_')
        self.coefficients[0] = intercept
        
    def _add_intercept(self, X):
        """Adds intercept to predictor array.
        """
        n_rows = X.shape[0]
        intercept = np.ones((n_rows, 1))
        return np.hstack([intercept, X])



