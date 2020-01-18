'''
Extend statsmodel families to improve IRLS initialization

IRLS will converge more quickly given reasonable initialization.
The following classes implement a custom IRLS initialization which
should offer a speed-up over the statsmodel default of starting with
all zeros
'''
import numpy as np
import statsmodels.api as sm

Gaussian = sm.families.family.Gaussian
InverseGaussian = sm.families.family.InverseGaussian
NegativeBinomial = sm.families.family.NegativeBinomial
Tweedie = sm.families.family.Tweedie
Binomial = sm.families.family.Binomial

def dummy_irls(*args, **kwargs):
    return None

for family in [
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Tweedie,
    Binomial]:
    family.irls_init = dummy_irls

class Poisson(sm.families.family.Poisson):
    '''Poisson GLM Class
    extends statsmodels.api.family classes to include
    smart initailization of IRLS'''

    def irls_init(self, X, y):
        '''Initialize IRLS Parameters

        Parameters
        ==========
        X: array-like
        2-D array of predictors, shape (n_obs, n_features)
        y: array-like
        array of response values of length n_obs

        Returns
        =======
        array of reasonable parameter estimates to start IRLS
        '''
        y_0 = np.log(y + 1e-4)
        params, _, _, _ = np.linalg.lstsq(X, y_0, rcond=None)
        return params

class Gamma(sm.families.family.Gamma):
    '''Gamma GLM Class
    extends statsmodels.api.family classes to include
    smart initailization of IRLS'''

    def irls_init(self, X, y):
        '''Initialize IRLS Parameters

        Parameters
        ==========
        X: array-like
        2-D array of predictors, shape (n_obs, n_features)
        y: array-like
        array of response values of length n_obs

        Returns
        =======
        array of reasonable parameter estimates to start IRLS
        '''
        y_0 = 1.0/np.maximum(y, 1e-4)
        params, _, _, _ = np.linalg.lstsq(X, y_0, rcond=None)
        return params


