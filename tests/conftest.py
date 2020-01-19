import pytest
from scipy.stats.distributions import poisson
import statsmodels.api as sm
import numpy as np

@pytest.fixture
def scotland_data():
    data = sm.datasets.scotland.load(as_pandas=False)
    return data.exog, data.endog

@pytest.fixture
def poisson_data():
    n_samples = 1000
    int_coef, age_coef, weight_coef = -10, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    expected_visits = np.exp(int_coef + age * age_coef + weight * weight_coef)
    observed_visits = poisson.rvs(expected_visits)
    X = np.vstack([age, weight]).T
    y = observed_visits
    return X, y

