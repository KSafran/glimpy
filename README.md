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
>>> from glimpy import GLM, Poisson
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
  
Now we can fit a `GLM` object to try to recover the formula we specified above
```python
>>> X = np.vstack([age, weight]).T
>>> y = observed_visits
>>> pglm = GLM(fit_intercept=True, family=Poisson)
>>> pglm.fit(X, y)
>>> print(pglm.summary())
                 Generalized Linear Model Regression Results
==============================================================================
Dep. Variable:                      y   No. Observations:                 1000
Model:                            GLM   Df Residuals:                      997
Model Family:                 Poisson   Df Model:                            2
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -3619.1
Date:                Thu, 09 Jan 2020   Deviance:                       967.43
Time:                        22:31:35   Pearson chi2:                     961.
No. Iterations:                     6
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const        -10.0132      0.020   -509.601      0.000     -10.052      -9.975
x1             0.0499      0.000    301.142      0.000       0.050       0.050
x2             0.0801      0.000    800.720      0.000       0.080       0.080
==============================================================================
```
