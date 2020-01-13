"""
Unit Testing GLMS
"""
import pytest
import numpy as np
from glimpy import (
    GLM,
    Poisson,
    Gamma,
    NegativeBinomial,
    InverseGaussian,
    Gaussian,
    Tweedie,
    Binomial,
)
import statsmodels.api as sm
from scipy.stats.distributions import poisson, nbinom, bernoulli, gamma

np.random.seed(10)

@pytest.fixture
def scotland_data():
    data = sm.datasets.scotland.load(as_pandas=False)
    return data.exog, data.endog


def test_gamma_example(scotland_data):
    """
    recreate the example here but through glimpy
    scikit-learn api
    https://www.statsmodels.org/stable/glm.html#module-reference
    """
    X, y = scotland_data
    gamma_glm = GLM(fit_intercept=True, family=Gamma())
    gamma_glm.fit(X, y)

    # SM Way
    X = sm.add_constant(X)
    sm_glm = sm.GLM(y, X, family=sm.families.Gamma())
    sm_glm = sm_glm.fit()
    assert np.all(np.isclose(sm_glm.params[1:], gamma_glm.coef_))
    assert np.isclose(sm_glm.params[0], gamma_glm.intercept_)


def test_poisson_example():
    """
    test a poisson model
    """
    n_samples = 1000
    int_coef, age_coef, weight_coef = -10, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    expected_visits = np.exp(int_coef + age * age_coef + weight * weight_coef)
    observed_visits = poisson.rvs(expected_visits)
    X = np.vstack([age, weight]).T
    y = observed_visits
    poisson_glm = GLM(fit_intercept=True, family=Poisson())
    poisson_glm.fit(X, y)
    assert np.all(np.isclose([age_coef, weight_coef], poisson_glm.coef_, rtol=1e-2))
    assert np.isclose(int_coef, poisson_glm.intercept_, rtol=1e-2)


def test_irls_init():
    """
    test that all distributions have irls initalizers
    """
    dists = [Gamma, Gaussian, InverseGaussian, Binomial, NegativeBinomial,
            Poisson]
    for dist in dists:
        assert hasattr(dist, 'irls_init')

def test_gaussian(scotland_data):
    '''
    test gaussian model
    '''
    X, y = scotland_data
    gauss_glm = GLM(fit_intercept=True, family=Gaussian())
    gauss_glm.fit(X, y)
    assert np.isclose(gauss_glm.intercept_, 137.414, rtol=1e-3)
    assert len(gauss_glm.coef_) == 7
    assert np.isclose(gauss_glm.coef_.sum(), -3.5915, rtol=1e-3)

def test_inverse_gaussian(scotland_data):
    '''
    test inverse gaussian model
    '''
    X, y = scotland_data
    igauss_glm = GLM(fit_intercept=True, family=InverseGaussian())
    igauss_glm.fit(X, y)
    assert np.isclose(igauss_glm.intercept_, -0.001072, rtol=1e-3)
    assert len(igauss_glm.coef_) == 7
    assert np.isclose(igauss_glm.coef_.sum(), 6.306e-5, rtol=1e-3)

def test_binomial():
    '''
    test binomial model
    '''
    n_samples = 2000
    int_coef, age_coef, weight_coef = -15, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    linear = (int_coef + age * age_coef + weight * weight_coef)
    prob_visit = 1.0/(1 + np.exp(-linear))
    visit = bernoulli.rvs(prob_visit)
    X = np.vstack([age, weight]).T
    y = visit
    binom_glm = GLM(fit_intercept=True, family=Binomial())
    binom_glm.fit(X, y)
    assert np.all(np.isclose([age_coef, weight_coef], binom_glm.coef_, rtol=2e-1))
    assert np.isclose(int_coef, binom_glm.intercept_, rtol=1e-1)

def test_tweedie():
    '''
    test tweedie model

    simulate tweedie data as poisson * gamma
    because tweedie isn't implemented in scipy
    or statsmodels yet
    '''
    n_samples = 1000
    int_coef, age_coef, weight_coef = -7, 0.02, 0.04
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    expected_visits = np.exp(int_coef + age * age_coef + weight * weight_coef)
    observed_visits = poisson.rvs(expected_visits)
    observed_cost = observed_visits * gamma.rvs(1000)
    X = np.vstack([age, weight]).T
    y = observed_cost
    tweedie_glm = GLM(fit_intercept=True, family=Tweedie())
    tweedie_glm.fit(X, y)
    assert np.all(np.isclose([age_coef, weight_coef], tweedie_glm.coef_, rtol=1e-1))

def test_neg_binomial():
    '''
    test negative binomial model
     mu = E(y) = ln(eta)
     mu proportional to p/(1-p)
    https://www.sagepub.com/sites/default/files/upm-binaries/21121_Chapter_15.pdf
    '''
    n_samples = 1000
    int_coef, age_coef, weight_coef = -15, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    linear = (int_coef + age * age_coef + weight * weight_coef)
    prob = 1.0/(1 + np.exp(linear))
    visits = nbinom.rvs(n=1, p=prob, size=n_samples)
    X = np.vstack([age, weight]).T
    y = visits
    nbinom_glm = GLM(fit_intercept=True, family=NegativeBinomial())
    nbinom_glm.fit(X, y)
    assert np.all(np.isclose([age_coef, weight_coef], nbinom_glm.coef_, rtol=1e-1))
    assert np.isclose(int_coef, nbinom_glm.intercept_, rtol=1e-1)
