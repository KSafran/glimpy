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

    penalty: string or None, default None
    one of ['l1', 'l2', 'elasticnet', None]
    norm to use in regularization penalty

    C: float > 0, default 1.0
    inverse of regularization strength - small values
    imply more regularization. Can also be an array
    matching the length of the number of model parameters.

    l1_ratio: 0 <= float <= 1.0, default 0.5
    elasticnet penalty ratio, only applies when
    penalty='elasticnet'. see
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    fit_intercept: bool, default=True
        whether to add an intercept column to X
    '''
    def __init__(self, family, penalty=None, C=1.0, l1_ratio=0.5, fit_intercept=True):
        self.family = family
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self._override_l1_ratio()
        self.fit_intercept = fit_intercept

    def _override_l1_ratio(self):
        '''Overrides l1 ratio if l1 or l2 set'''
        if self.penalty == 'l1':
            self.l1_ratio = 1.0
        elif self.penalty == 'l2':
            self.l2_ratio = 0.0
        return self

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
        if self.penalty is None:
            self.glm = self.glm.fit(start_params=self.family.irls_init(X, y))
        else:
            self.glm = self.glm.fit_regularized(method='elasticnet',
                alpha=1.0/self.C,
                start_params=self.family.irls_init(X, y),
                L1_wt=self.l1_ratio)
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
