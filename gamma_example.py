from scipy.stats.distributions import gamma
import numpy as np
from glimpy.gamma import GammaGLM

# for gamma glm mu = XB = alpha * scale and scale is a constant
test_scale = 2

x_1 = np.random.uniform(0, 1, 1000)
x_2 = np.random.uniform(0, 1, 1000)

x_eta = 1 + 2 * x_1 - x_2

y_obs = gamma.rvs(test_scale, scale=1.0/x_eta)

X = np.vstack([np.ones(len(x_1)), x_1, x_2]).T
y = y_obs.reshape(-1, 1)

gglm = GammaGLM(fit_intercept=False)
gglm.fit(X, y)

