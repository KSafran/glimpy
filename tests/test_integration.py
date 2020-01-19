'''test integration with sklearn api'''
import numpy as np
from scipy.stats.distributions import poisson, nbinom, bernoulli, gamma
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

np.random.seed(10)

def test_cross_val():
    """
    test sklearn cross validation
    """
    n_samples = 100
    int_coef, age_coef, weight_coef = -10, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    expected_visits = np.exp(int_coef + age * age_coef + weight * weight_coef)
    observed_visits = poisson.rvs(expected_visits)
    X = np.vstack([age, weight]).T
    y = observed_visits
    poisson_glm = GLM(fit_intercept=True, family=Poisson())
    poisson_glm.fit(X, y)
    cv_results = cross_val_score(poisson_glm, X, y, cv=2)
    assert len(cv_results) == 2

def test_pipeline():
    """
    test sklearn pipelines
    """
    n_samples = 100
    int_coef, age_coef, weight_coef = -10, 0.05, 0.08
    age = np.random.uniform(30, 70, n_samples)
    weight = np.random.normal(150, 20, n_samples)
    expected_visits = np.exp(int_coef + age * age_coef + weight * weight_coef)
    observed_visits = poisson.rvs(expected_visits)
    X = np.vstack([age, weight]).T
    y = observed_visits
    scaler = StandardScaler()
    poisson_glm = GLM(fit_intercept=True, family=Poisson())
    poisson_pipe = Pipeline([('scaler', scaler), ('glm', poisson_glm)])
    poisson_pipe.fit(X, y)
    preds = poisson_pipe.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 100

