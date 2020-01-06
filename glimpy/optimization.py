'''Optimization for GLMs'''
import numpy as np

def poisson_irls(X, y, max_iter=100, tolerance=1e-4):
    '''Iteratively Reweighted Least Squares for Poisson GLM

    Algorithm described in detail in section 4.2.1 in this text
    https://data.princeton.edu/wws509/notes/c4.pdf
    '''
    # initialize
    y_0 = np.log(y + 1e-4)
    beta_ = weighted_ols(X, y_0)

    for iter_ in range(max_iter):
        eta_ = X @ beta_
        w_ = np.eye(X.shape[0]) * np.exp(eta_)
        z_ = working_dependent(X, y, beta_)
        beta_new = weighted_ols(X, z_, w_)
        if np.abs(beta_new - beta_).max() < tolerance:
            break
        beta_ = beta_new

    if iter_ == (max_iter - 1):
        print('failed to converge after {max_iter} iterations')
    else:
        print(f'converged after {iter_} iterations')
    return beta_new


def weighted_ols(X, y, W=None):
    '''OLS closed form solution'''
    if W is None:
        W = np.eye(X.shape[0])
    betas = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    return betas

def working_dependent(X, y, beta):
    '''Working depending variable for Poisson IRLS'''
    eta = X @ beta
    return eta + (y - np.exp(eta))/np.exp(eta)
