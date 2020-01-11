'''
Unit Testing GLMS
'''
import numpy as np
from glimpy import GLM, Poisson, Gamma
import statsmodels.api as sm
from scipy.stats.distributions import poisson
np.random.seed(10)

def test_gamma_example():
    '''
    recreate the example here but through glimpy
    scikit-learn api
    https://www.statsmodels.org/stable/glm.html#module-reference
    '''
    data = sm.datasets.scotland.load(as_pandas=False)
    X = data.exog
    y = data.endog
    gamma_glm = GLM(fit_intercept=True, family=Gamma)
    gamma_glm.fit(X, y)

    # SM Way
    data.exog = sm.add_constant(data.exog)
    sm_glm = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
    sm_glm = sm_glm.fit()
    assert np.all(np.isclose(sm_glm.params[1:], gamma_glm.coef_))
    assert np.isclose(sm_glm.params[0], gamma_glm.intercept_)

def test_poisson_example():
    '''
    test a poisson model
    '''
    n_samples = 1000
    int_coef, age_coef, weight_coef = -10, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    expected_visits = np.exp(int_coef + age * age_coef + weight * weight_coef)
    observed_visits = poisson.rvs(expected_visits)
    X = np.vstack([age, weight]).T
    y = observed_visits
    poisson_glm = GLM(fit_intercept=True, family=Poisson)
    poisson_glm.fit(X, y)
    assert np.all(np.isclose([age_coef, weight_coef], poisson_glm.coef_, rtol=1e-2))
    assert np.isclose(int_coef, poisson_glm.intercept_, rtol=1e-2)

