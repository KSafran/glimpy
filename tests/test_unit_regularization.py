'''test regularization'''
import numpy as np
from glimpy import GLM, Poisson

def test_l1_poisson(poisson_data):
    """
    test a poisson model
    """
    X, y = poisson_data
    poisson_glm = GLM(fit_intercept=True, family=Poisson(),
        penalty='l1', C=0.0001)
    poisson_glm.fit(X, y)
    assert poisson_glm.coef_[0] == 0
    assert np.abs(poisson_glm.intercept_) > 1
    assert poisson_glm.l1_ratio == 1

def test_l2_poisson(poisson_data):
    """
    test a poisson model with l2 regularization
    """
    X, y = poisson_data
    poisson_glm = GLM(fit_intercept=False, family=Poisson(),
        penalty='l2', C=0.001)
    intercept_col = np.ones(len(X)).reshape(-1, 1)
    X = np.hstack([intercept_col, X])
    poisson_glm.fit(X, y)
    assert np.all(np.round(poisson_glm.coef_, 2) == [-0.03, 0.01, 0.03])
    assert poisson_glm.l1_ratio == 0

def test_elasticnet_poisson(poisson_data):
    """
    test a poisson model with l2 regularization
    """
    X, y = poisson_data
    poisson_glm = GLM(fit_intercept=False, family=Poisson(),
        penalty='elasticnet', C=0.001)
    intercept_col = np.ones(len(X)).reshape(-1, 1)
    X = np.hstack([intercept_col, X])
    poisson_glm.fit(X, y)
    assert np.all(np.round(poisson_glm.coef_, 2) == [0, 0, 0.04])
    assert poisson_glm.l1_ratio == 0.5

