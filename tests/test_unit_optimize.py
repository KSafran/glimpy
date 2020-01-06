"""Test optimization code"""
import numpy as np
from scipy.stats import poisson
from glimpy.optimization import poisson_irls, weighted_ols, poisson_z

def test_weighted_ols():
    '''test weighted_ols works as expected'''
    X = np.eye(2)
    y = np.array([[1], [2]])
    betas = weighted_ols(X, y)
    assert np.all(betas == np.array([[1], [2]]))

    # test weighted version
    X = np.array([[1], [1]])
    y = np.array([[1], [2]])
    W = np.array([[1, 0], [0, 3]])
    betas = weighted_ols(X, y, W)
    assert betas[0][0] == 1.75

def test_poisson_z():
    X = np.eye(2)
    y = np.array([[1], [2]])
    betas = np.array([[1], [2]])
    dep = poisson_z(X, y, betas)
    assert dep[0][0] == (1 + (1 - np.exp(1))/np.exp(1))
    assert dep[1][0] == (2 + (2 - np.exp(2))/np.exp(2))

def test_poisson_irls():
    np.random.seed(10)
    n_samples = 1000
    x_1 = np.random.uniform(0, 1, n_samples)
    x_2 = np.random.uniform(0, 1, n_samples)
    lam = np.exp(2 + 3 * x_1 - 2 * x_2)
    observed = poisson.rvs(lam)

    X = np.vstack([np.ones(n_samples), x_1, x_2]).T
    y = observed.reshape(-1, 1)

    results = poisson_irls(X, y)
    assert np.abs(results[:, 0] - [2, 3, -2]).max() < 0.1


