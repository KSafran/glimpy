'''Optimization for GLMs'''
import numpy as np
from .link import logit, anti_logit


def irls_constructor(initialization, z_function, w_function):
    '''Iteratively Reweighted Least Squares Constructor

    Algorithm described in detail in section 4.2.1 in this text
    https://data.princeton.edu/wws509/notes/c4.pdf
    '''
    def fitter(X, y, max_iter=100, tolerance=1e-4):
        # initialize
        beta_ = initialization(X, y)

        for iter_ in range(max_iter):
            w_ = w_function(X, beta_)
            z_ = z_function(X, y, beta_)
            beta_new = weighted_ols(X, z_, w_)
            if np.abs(beta_new - beta_).max() < tolerance:
                break
            beta_ = beta_new

        if iter_ == (max_iter - 1):
            print('failed to converge after {max_iter} iterations')
        else:
            print(f'converged after {iter_} iterations')
        return beta_new
    return fitter

def weighted_ols(X, y, W=None):
    '''OLS closed form solution'''
    if W is None:
        W = np.eye(X.shape[0])
    betas = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    return betas

# Poisson
# https://data.princeton.edu/wws509/notes/c4.pdf
def poisson_z(X, y, beta):
    '''Working depending variable for Poisson IRLS'''
    eta = X @ beta
    return eta + (y - np.exp(eta))/np.exp(eta)

def poisson_w(X, beta):
    '''Poisson IRLS function for W'''
    eta = X @ beta
    return np.eye(X.shape[0]) * np.exp(eta)

def poisson_beta_init(X, y):
    '''Poisson IRLS initial beta estimate'''
    y_0 = np.log(y + 1e-4)
    return weighted_ols(X, y_0)

# Bernoilli
# https://data.princeton.edu/wws509/notes/c3.pdf
# n = binomial denominator = 1 for bernoulli
def bernoulli_z(X, y, beta):
    '''Working depending variable for bernoulli IRLS'''
    eta = X @ beta
    mu = anti_logit(eta)
    return eta + (y - mu)/(mu * (1 - mu))

def bernoulli_w(X, beta):
    '''bernoulli IRLS function for W'''
    eta = X @ beta
    mu = anti_logit(eta)
    return np.eye(X.shape[0]) * anti_logit(eta) * (1 - mu)

def bernoulli_beta_init(X, y):
    '''bernoulli IRLS initial beta estimate'''
    z = np.log((y + 0.5)/(1 - y + 0.5))
    return weighted_ols(X, z)

bernoulli_irls = irls_constructor(bernoulli_beta_init, bernoulli_z, bernoulli_w)
poisson_irls = irls_constructor(poisson_beta_init, poisson_z, poisson_w)
