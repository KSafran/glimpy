# Glimpy
[![CircleCI](https://circleci.com/gh/KSafran/glimpy.svg?style=svg)](https://circleci.com/gh/KSafran/glimpy)  

glimpy is a Python module for fitting generalized linear models. It's based on the [scikit-learn](https://scikit-learn.org/stable/index.html) API to facilitate use with other scikit-learn tools (pipelines, cross-validation, etc.).

## Installation

## Getting Started
Here is an example of a poisson GLM to help get you started

```python
>>> import numpy as np
>>> from scipy.stats import poisson
>>> from glimpy.poisson import PoissonGLM
>>>
>>> np.random.seed(10)
>>> n_samples = 1000
>>>
>>> # patients ages range from 30 to 70
... age = np.random.uniform(30, 70, n_samples)
>>> # patients wieghts have mean 150 sd of 20
... weight = np.random.normal(150, 20, n_samples)
>>>
>>> # simulate known relationship between age, weight and expected
... # number of hospital visits
... expected_visits = np.exp(-10 + age * 0.05 + weight * 0.08)
>>> observed_visits = poisson.rvs(expected_visits)
>>>
>>> # Fit our Poisson GLM and observe the results
... X = np.vstack([age, weight]).T
>>> y = observed_visits.reshape(-1, 1)
>>> pglm = PoissonGLM(fit_intercept=True)
>>> pglm.fit(X, y)
Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: -2024.688881
         Iterations: 62
         Function evaluations: 388
         Gradient evaluations: 377
>>> print(pglm.coefficients)
[-10.01316558   0.04992275   0.08008428]
```

