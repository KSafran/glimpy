"""Base GLM Class"""
import numpy as np
from sklearn.base import BaseEstimator
import statsmodels.api as sm


class GLM(BaseEstimator):
    '''Base GLM Class

    Extends the scikit-learn BaseEstimator class which allows
    for simple interaction with scikit-learn API. Properties
    for coef_ and intercept_ attributes allow the classes to
    conform to the scikit-learn API while allowing the
    implementation to store all model coefficients together
    in a single `coefficients` attribute.

    Models are fit using statsmodels implementation of
    iteratively reweighted least squares. See the
    statsmodels documentation for technical details
    https://www.statsmodels.org/stable/glm.html

    Parameters
    ==========
    family: statsmodels.family object, required
    https://www.statsmodels.org/stable/glm.html#families

    link: sm.families.links object, default None
    Uses the canonical link if none is specified
    not all links available for each class. see
    https://www.statsmodels.org/stable/glm.html#families
    for details

    fit_intercept: bool, default=True
        whether to add an intercept column to X
    '''
    def __init__(self, family, link=None, fit_intercept=True):
        # self.family = family(link)
        self.family = family
        self.fit_intercept = fit_intercept

    @property
    def coef_(self):
        '''An array of coefficients, excludes the intercept.'''
        if self.fit_intercept:
            return self.glm.params[1:]
        return self.glm.params

    @coef_.setter
    def coef_(self, coef):
        '''Sets Model Coefficients

        This assumes you will also set the intercept
        '''
        if coef.shape != self.glm.params:
            raise ValueError("coef shape does not match")
        self.glm.params = coef

    @property
    def intercept_(self):
        '''Model Intercept Parameter

        Assumes model fit with intercept, otherwise
        returns None
        '''
        if self.fit_intercept:
            return self.glm.params[0]
        return None

    @intercept_.setter
    def intercept_(self, intercept):
        '''Sets Model Intercept

        Only allowed if fit with fit_intercept=True
        '''
        if not self.fit_intercept:
            raise AttributeError('not fit with intercept_')
        self.glm.params[0] = intercept

    def _add_intercept(self, X):
        """Adds intercept to predictor array.
        """
        n_rows = X.shape[0]
        intercept = np.ones((n_rows, 1))
        return np.hstack([intercept, X])

    def fit(self, X, y, sample_weight=None, offset=None):
        """Fits a poisson glm using bfgs

        Parameters
        ==========
        X: array-like
        2-D array of predictors, shape (n_obs, n_features)
        y: array-like
        array of response values of length n_obs
        offset: array-like, optional
        a 1-D array of offset values
        sample_weights: array-like, optional
        a 1-D array of weight values
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.glm = sm.GLM(y, X, family=self.family, offset=offset, freq_weights=sample_weight)
        self.glm = self.glm.fit(start_params=self.family.irls_init(X, y))
        return self

    def predict(self, X):
        '''Predicts using fitted GLM

        Parameters
        ==========
        X: array-like
        2-D array of predictors

        Returns
        =======
        A 1-D array of predicted values
        '''
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self.glm.predict(X)

    def score(self, X, y, sample_weight=None, score_fun='deviance'):
        '''Return the deviance for a fitted GLM

        X: array-like
        2-D array of predictors, shape (n_obs, n_features)
        y: array-like
        array of response values of length n_obs
        offset: array-like, optional
        a 1-D array of offset values
        sample_weights: array-like, optional
        a 1-D array of weight values
        score_fun: string, default='deviance
        what score to return, either 'deviance' to
        return deviance or 'nll' to return negative log
        likelihood

        Returns
        =======
        float of model deviance (or negative log likelihood)
        '''
        fitted = self.predict(X)
        if sample_weight is None:
            sample_weight = 1
        if score_fun == 'deviance':
            score = self.family.deviance(y, fitted, freq_weights=sample_weight)
        elif score_fun.lower() == 'nll':
            score = -self.family.loglike(y, fitted, freq_weights=sample_weight)
        else:
            raise ValueError('score_fun not an accepted scoring function')
        return score

    def summary(self):
        '''
        Return a summary from statsmodels
        '''
        return self.glm.summary()
