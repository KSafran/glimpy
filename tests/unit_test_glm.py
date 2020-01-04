import numpy as np
from scipy.stats import poisson
from scipy.special import factorial
from pyglm.glm import poisson_score, poisson_score_grad, PoissonGLM

def test_poisson_score():
    '''
    poisson score should give us negative log likelihood
    but without the term that doesn't depend on thetas (log(y!))
    https://en.wikipedia.org/wiki/Poisson_regression
    '''
    thetas = np.array([1])
    x = np.array([[2], [2]])
    y = np.array([[3], [3]])
    p_score = poisson_score(x=x, y=y, thetas=thetas)

    # calculate score by hand
    lam = np.exp(thetas * x)
    log_likelihood = poisson.logpmf(y, lam)
    score = -np.mean(log_likelihood + np.log(factorial(y)))
    assert np.isclose(p_score, 
        score)
    
def test_poisson_grad():
    '''
    gradient of scoring function is used for gradient descent
    optimization algorithms. gradients should be close to 
    0 at optimums
    '''
    thetas = np.array([1])
    x = np.array([np.log(2)])
    y = np.array([2])
    score_grad = poisson_score_grad(x, y, thetas)
    assert np.isclose(score_grad, 0)
    
def test_poisson_glm():
    '''
    test that it finds optimum for an easy problem
    '''
    X = np.array([[1, 0], [1, 1]])
    y = np.array([[1], [2]])
    pglm = PoissonGLM(fit_intercept=False)
    pglm.fit(X, y)
    assert np.isclose(pglm.coef_[0], 0, atol=1e-6)
    assert np.isclose(pglm.coef_[1], np.log(2))

    # Test Prediction
    preds = pglm.predict(X)
    assert np.isclose(preds[0], 1)
    assert np.isclose(preds[1], 2)

    # Test Scoring
    score = pglm.score(X, y)
    assert score == poisson_score(X, y, pglm.coef_)
    assert pglm.intercept_ is None

def test_poisson_glm_intercept():
    '''
    test that it finds optimum for an easy problem
    '''
    X = np.array([[0], [1]])
    y = np.array([[1], [2]])
    pglm = PoissonGLM()
    pglm.fit(X, y)
    assert np.isclose(pglm.intercept_, 0, atol=1e-6)
    assert np.isclose(pglm.coef_[0], np.log(2))

    # Test Prediction
    preds = pglm.predict(X)
    assert np.isclose(preds[0], 1)
    assert np.isclose(preds[1], 2)

    # Test Scoring
    score = pglm.score(X, y)
    X_i = pglm._add_intercept(X)
    assert np.isclose(score, poisson_score(X_i, y, np.array([0, np.log(2)])))







