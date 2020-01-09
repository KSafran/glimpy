# Glimpy
[![CircleCI](https://circleci.com/gh/KSafran/glimpy.svg?style=svg)](https://circleci.com/gh/KSafran/glimpy)  

glimpy is a Python module for fitting generalized linear models. It's based on the [scikit-learn](https://scikit-learn.org/stable/index.html) API to facilitate use with other scikit-learn tools (pipelines, cross-validation, etc.). Models are fit using the [statsmodels](https://www.statsmodels.org/stable/glm.html) package.

## Installation
`pip install git+https://github.com/KSafran/glimpy`

## Getting Started
Here is an example of a poisson GLM to help get you started

We will simulate an experiment where we want to determine how an individual's age and weight influence the number of hospital visits they can expect to have in a given year.  

Start with basic imports and setup 
```python
>>> import numpy as np
>>> from scipy.stats import poisson
>>> from glimpy import PoissonGLM
>>>
>>> np.random.seed(10)
>>> n_samples = 1000
```
  
Now we will simulate some data where observed individuals have ages ranging from 30 to 70, and weights normally distributed centered around 150 lbs.
```python  
>>> age = np.random.uniform(30, 70, n_samples)
>>> weight = np.random.normal(150, 20, n_samples)
```
  
Then we will have the expected number of hospital visits vary according to the following equation. We will sample from a poisson distribution with those means to get a sample of observed hospital visits
```python
>>> expected_visits = np.exp(-10 + age * 0.05 + weight * 0.08)
>>> observed_visits = poisson.rvs(expected_visits)
```
  
Now we can fit a `PoissonGLM` object to try to recover the formula we specified above
```python
>>> X = np.vstack([age, weight]).T
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

