'''Optimization for GLMs'''
import numpy as np
from .link import logit, anti_logit, inverse


def irls_constructor(initialization, z_function, w_function):
    '''Iteratively Reweighted Least Squares Constructor

    Algorithm described in detail in section 4.2.1 in this text
    https://data.princeton.edu/wws509/notes/c4.pdf
    '''
    def fitter(X, y, max_iter=100, tolerance=1e-4):
        # initialize
        beta_ = initialization(X, y)

        for iter_ in range(max_iter):
            print(beta_)
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

# Gamma
# https://data.princeton.edu/wws509/notes/a2.pdf
# z = eta + (y - mu) d_eta/d_mu
# where d# z = eta + (y - mu) d_eta/d_mu
# where d_eta/d_mu is derivative of the link function
# gamma inverse link has derivative -1/(x**2)
# evaluated at trial estimate
# w = p/[b"(theta) * (d_eta/d_mu)**2]
# where b"(theta) is the second derivative of
# B(theta) = -log(-theta) so
# http://people.stat.sfu.ca/~raltman/stat402/402L25.pdf


def gamma_z(X, y, beta):
    '''Working depending variable for gamma IRLS'''
    eta = X @ beta
    mu = inverse(eta)
    dmu_deta = -1.0 / (eta ** 2)
    blah = eta + ((y - mu) / dmu_deta)
    breakpoint()
    return blah

def gamma_w(X, beta):
    '''gamma IRLS function for W'''
    eta = X @ beta
    mu = inverse(eta)
    dmu_deta = -1.0 / (eta ** 2)
    blah = np.eye(X.shape[0]) * np.sqrt((dmu_deta ** 2) / (mu ** 2))
    return blah

def gamma_beta_init(X, y):
    '''gamma IRLS initial beta estimate'''
    z = inverse(y)
    return weighted_ols(X, z)

bernoulli_irls = irls_constructor(bernoulli_beta_init, bernoulli_z, bernoulli_w)
poisson_irls = irls_constructor(poisson_beta_init, poisson_z, poisson_w)
gamma_irls = irls_constructor(gamma_beta_init, gamma_z, gamma_w)
