'''Scoring Functions for GLMs '''
import numpy as np

def poisson_score(X, y, betas):
    """Scores Poisson Model

    Scores a simplified version of negative log likilihood that 
    ignores terms that doesn't depend on model params (i.e. log(y!))

    Parameters
    ==========
    X: np.ndarray of predictors, shape (n_obs, n_features)
    y: np.ndarray response values, shape (n_obs, 1)
    betas: np.ndarray of coefficients, shape (n_features)

    Returns
    =======
    Model score, float
    """
    betas = betas.reshape(-1, 1)
    score_i = np.exp(X @ betas) - (y * (X @ betas))
    return np.mean(score_i)


def poisson_score_grad(X, y, betas):
    """Gradient of Poisson Model Score

    Computes gradient of poisson score with respect to the
    model parameters (betas)

    Parameters
    ==========
    X: np.ndarray of predictors, shape (n_obs, n_features)
    y: np.ndarray response values, shape (n_obs, 1)
    betas: np.ndarray of coefficients, shape (n_features)

    Returns
    =======
    np.array of gradients, shape (n_features)
    """
    betas = betas.reshape(-1, 1)
    grad_i = (X * np.exp(X @ betas)) - (y * X)
    return np.sum(grad_i, axis=0)


def poisson_deviance(X, y, betas):
    """Calculates Poisson Model Deviance

    Deviance = 2 * sum(log likelihood saturdated - log likelihood model)
    where the saturated model is the case where mu_i=y_i
    https://data.princeton.edu/wws509/notes/a2s5


    Parameters
    ==========
    X: np.ndarray of predictors, shape (n_obs, n_features)
    y: np.ndarray response values, shape (n_obs, 1)
    betas: np.ndarray of coefficients, shape (n_features)

    Returns
    =======
    Model deviance, float
    """
    mu = np.exp(X @ betas.reshape(-1, 1))
    score_i = y * np.log(y/mu) - (y - mu)
    return 2 * (score_i).sum()

def gamma_inverse_score(X, y, shape, betas):
    """Scores Poisson Model

    Scores a simplified version of negative log likilihood that 
    ignores terms that doesn't depend on model params. This asssumes
    the shape parameter is constant across all observations.

    Parameters
    ==========
    X: np.ndarray of predictors, shape (n_obs, n_features)
    y: np.ndarray response values, shape (n_obs, 1)
    shape: value of shape parameter, positive float
    betas: np.ndarray of coefficients, shape (n_features)

    Returns
    =======
    Model score, float
    """
    link = lambda x: 1.0/x
    xb = X @ betas.reshape(-1, 1)
    lam = link(np.maximum(1e-4, np.abs(xb))) #TODO is this good enough?
    score_i =  (y / lam) + (shape * np.log(lam))
    return np.mean(score_i)

# def gamma_inverse_score_grad(X, y, shape, betas):
#     """
#     simplified version of negative log likilihood, 
#     we hold the shape parameter constant and vary
#     the scale parameters

#     likelihood formula can be found here, use shape, scale 
#     parameterization
#     https://en.wikipedia.org/wiki/Gamma_distribution#Closed-form_estimators
#     we are holding shape constant and predicting lambda so
#     we can ignore terms without lambda

#     X: two dimensional np.ndarray of predictors
#     y: two dimensional np.ndarray of response values
#     shape: value of the shape parameter used in scoring
#     betas: one dimensional nd.array of coefficients
#     """
#     grad_i = X * y - shape/betas
#     breakpoint()
#     return np.mean(grad_i)
