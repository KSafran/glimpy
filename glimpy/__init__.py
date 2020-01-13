'''Generalized Linear Models in Python

glimpy is a Python module for fitting generalized linear models. It
aims to follow the scikit-learn API closely enough to improve ease of
use and to take advantage of useful scikit-learn tools such as
`sklearn.pipeline.Pipeline` and `sklearn.model_selection.cross_val_score`.
'''
from .glm import GLM
from .families import (
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Tweedie,
    Binomial,
    Gamma,
    Poisson
    )

__all__ = ["GLM", "Gaussian", "InverseGaussian", "NegativeBinomial", "Tweedie",
    "Binomial", "Gamma", "Poisson"]
