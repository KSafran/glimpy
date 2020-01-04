''' scoring functions for GLMs '''
import numpy as np

def poisson_score(X, y, thetas):
    """
    simplified version of negative log likilihood, 
    ignore term that doesn't depend on model params (log(y!))

    X: two dimensional np.ndarray of predictors
    y: two dimensional np.ndarray of response values
    thetas: one dimensional nd.array of coefficients
    """
    thetas = thetas.reshape(-1, 1)
    score_i = np.exp(X @ thetas) - (y * (X @ thetas))
    return np.mean(score_i)


def poisson_score_grad(X, y, thetas):
    """
    partial derivatives of score with respect to thetas

    X: two dimensional np.ndarray of predictors
    y: two dimensional np.ndarray of response values
    thetas: one dimensional nd.array of coefficients
    """
    thetas = thetas.reshape(-1, 1)
    grad_i = (X * np.exp(X @ thetas)) - (y * X)
    return np.sum(grad_i, axis=0)


