'''Link functions for GLMS'''
import numpy as np

def inverse(x):
    return 1.0/x

def log(x):
    return np.log(x)

def logit(x):
    return np.log(x/(1 - x))

def identity(x):
    return x

def anti_logit(x):
    return 1/(1 + np.exp(-x))
