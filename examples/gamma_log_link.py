'''Gamma GLM Example'''
import statsmodels.api as sm
from glimpy import GLM, Gamma

data = sm.datasets.scotland.load(as_pandas=False)
X = data.exog
y = data.endog
gamma_glm = GLM(fit_intercept=True, family=Gamma, link=sm.families.links.log)
gamma_glm.fit(X, y)
print(gamma_glm.summary())
