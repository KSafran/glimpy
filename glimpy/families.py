'''
Extend statsmodel families to improve IRLS initialization

IRLS will converge more quickly given reasonable initialization.
The following classes implement a custom IRLS initialization which
should offer a speed-up over the statsmodel default of starting with
all zeros
'''
from statsmodels.api.families import (Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Poisson,
    Tweedie)

class Gamma(Gamma)
