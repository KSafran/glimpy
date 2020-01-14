'''Gamma GLM Example'''
import statsmodels.api as sm
from glimpy import GLM, Gamma
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = sm.datasets.scotland.load(as_pandas=False)
X = data.exog
y = data.endog

scaler = StandardScaler()
gamma_glm = GLM(fit_intercept=True, family=Gamma(link=sm.families.links.log))
gamma_pipeline = Pipeline([('scaler', scaler), ('glm', gamma_glm)])
gamma_pipeline.fit(X, y)
print(gamma_glm.summary())
