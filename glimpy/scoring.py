''' scoring functions for GLMs '''
import numpy as np

def poisson_score(X, y, betas):
    """
    simplified version of negative log likilihood, 
    ignore term that doesn't depend on model params (log(y!))

    X: two dimensional np.ndarray of predictors
    y: two dimensional np.ndarray of response values
    betas: one dimensional nd.array of coefficients
    """
    betas = betas.reshape(-1, 1)
    score_i = np.exp(X @ betas) - (y * (X @ betas))
    return np.mean(score_i)


def poisson_score_grad(X, y, betas):
    """
    partial derivatives of score with respect to betas

    X: two dimensional np.ndarray of predictors
    y: two dimensional np.ndarray of response values
    betas: one dimensional nd.array of coefficients
    """
    betas = betas.reshape(-1, 1)
    grad_i = (X * np.exp(X @ betas)) - (y * X)
    return np.sum(grad_i, axis=0)


def gamma_inverse_score(X, y, shape, betas):
    """
    simplified version of negative log likilihood, 
    we hold the shape parameter constant and vary
    the scale parameters

    log-likelihood formula can be found here, use shape, scale
    parameterization
    https://en.wikipedia.org/wiki/Gamma_distribution
    we are holding shape constant and predicting lambda so
    we can ignore terms without lambda

    X: two dimensional np.ndarray of predictors
    y: two dimensional np.ndarray of response values
    shape: value of the shape parameter used in scoring
    betas: one dimensional nd.array of coefficients
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
