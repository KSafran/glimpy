'''
an example of a poisson glm in action

we will simulate some data and model the number of hospital
trips for a patient in one year as a function of the patient's
age and weight
'''
import numpy as np
from scipy.stats import poisson
from glimpy.poisson import PoissonGLM

np.random.seed(10)
n_samples = 1000

# patients ages range from 30 to 70
age = np.random.uniform(30, 70, n_samples)
# patients wieghts have mean 150 sd of 20
weight = np.random.normal(150, 20, n_samples)

# simulate known relationship between age, weight and expected
# number of hospital visits
expected_visits = np.exp(-10 + age * 0.05 + weight * 0.08)
observed_visits = poisson.rvs(expected_visits)

# Fit our Poisson GLM and observe the results
X = np.vstack([age, weight]).T
y = observed_visits.reshape(-1, 1)
pglm = PoissonGLM(fit_intercept=True)
pglm.fit(X, y)
print(pglm.coefficients)

