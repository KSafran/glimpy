import numpy as np
from scipy.stats import poisson, gamma
from scipy.special import factorial
from glimpy.scoring import poisson_deviance
from glimpy.poisson import poisson_score, poisson_score_grad, PoissonGLM
from glimpy.gamma import GammaGLM, gamma_inverse_score #gamma_inverse_score_grad
from glimpy.normal import NormalGLM


def test_poisson_score():
    """
    poisson score should give us negative log likelihood
    but without the term that doesn't depend on betas (log(y!))
    https://en.wikipedia.org/wiki/Poisson_regression
    """
    betas = np.array([1])
    x = np.array([[2], [2]])
    y = np.array([[3], [3]])
    p_score = poisson_score(X=x, y=y, betas=betas)

    # calculate score by hand
    lam = np.exp(betas * x)
    log_likelihood = poisson.logpmf(y, lam)
    score = -np.mean(log_likelihood + np.log(factorial(y)))
    assert np.isclose(p_score, score)

def test_poisson_deviance():
    """
    poisson deviance should be zero when our model is equal
    to the saturated model
    https://data.princeton.edu/wws509/notes/a2s5
    """
    betas = np.array([1])
    x = np.array([[np.log(2)], [np.log(3)]])
    y = np.array([[2], [3]])
    p_deviance = poisson_deviance(X=x, y=y, betas=betas)
    assert np.isclose(p_deviance, 0)

def test_poisson_grad():
    """
    gradient of scoring function is used for gradient descent
    optimization algorithms. gradients should be close to 
    0 at optimum
    """
    betas = np.array([np.log(2), np.log(3)])
    x = np.eye(2)
    y = np.array([[2], [3]])
    score_grad = poisson_score_grad(x, y, betas)
    assert score_grad.shape == betas.shape
    assert np.all(np.isclose(score_grad, 0))

    # too high
    score_grad = poisson_score_grad(x, y, betas + 1)
    assert np.all(score_grad > 0)

    # too low
    score_grad = poisson_score_grad(x, y, betas - 1)
    assert np.all(score_grad < 0)


def test_poisson_glm():
    """
    test that it finds optimum for an easy problem
    """
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
    """
    test that it finds optimum for an easy problem
    """
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

def test_gamma_score():
    """
    gamma score should give us something similar to 
    negative log likelihood, just without parts that dont depend 
    on the scale parameter

    http://statweb.stanford.edu/~susan/courses/s200/lectures/lect11.pdf
    """
    betas = np.array([1])
    x = np.array([[2], [2]])
    y = np.array([[3], [3]])
    g_score = gamma_inverse_score(X=x, y=y, shape=1, betas=betas)

    # calculate score by hand
    lam = 1.0/(betas * x)
    log_likelihood = gamma.logpdf(y, scale=lam, a=1)
    # when shape is 1 the other terms disappear
    # (shape - 1)*log(y) and log(gamma(1))
    score = -np.mean(log_likelihood)
    assert np.isclose(g_score, score)

# def test_gamma_grad():
#     """
#     gradient of scoring function is used for gradient descent
#     optimization algorithms. gradients should be close to 
#     0 at optimum
#     """
#     betas = np.array([0.5, 0.25])
#     x = np.array([[1, 0], [0, 1]])
#     y = np.array([[2],[4]])
#     score_grad = gamma_inverse_score_grad(x, y, shape=1, betas=betas)
#     assert np.isclose(score_grad, 0)

def test_gamma_glm():
    """
    test that it finds optimum for an easy problem
    """
    X = np.array([[1, 0], [0, 1]])
    y = np.array([[1], [2]])
    gglm = GammaGLM(fit_intercept=False, shape=1)
    gglm.fit(X, y)
    assert np.isclose(gglm.coef_[0], 1, atol=1e-6)
    assert np.isclose(gglm.coef_[1], 0.5)

    # Test Prediction
    preds = gglm.predict(X)
    assert np.isclose(preds[0], 1)
    assert np.isclose(preds[1], 2)

    # Test Scoring
    score = gglm.score(X, y)
    assert score == gamma_inverse_score(X, y, 1, gglm.coef_)
    assert gglm.intercept_ is None


def test_normal_glm():
    """
    test that it finds optimum for an easy problem
    """
    X = np.array([[1, 0], [0, 1]])
    y = np.array([[1], [2]])
    nglm = NormalGLM(fit_intercept=False)
    nglm.fit(X, y)
    assert np.isclose(nglm.coef_[0], 1)
    assert np.isclose(nglm.coef_[1], 2)

    # Test Prediction
    preds = nglm.predict(X)
    assert np.isclose(preds[0], 1)
    assert np.isclose(preds[1], 2)

    # Test Scoring
    score = nglm.score(X, y)
    assert np.isclose(score, 0)
    assert nglm.intercept_ is None

