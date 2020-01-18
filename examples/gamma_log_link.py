'''Gamma GLM Example'''
import statsmodels.api as sm
from glimpy import GLM, Gamma
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

diabetes = datasets.load_diabetes()

scaler = StandardScaler()
gamma_glm = GLM(fit_intercept=True, family=Gamma(sm.families.links.log()), penalty='elasticnet')
gamma_pipeline = Pipeline([('scaler', scaler), ('glm', gamma_glm)])
grid_search = GridSearchCV(gamma_pipeline,
    param_grid=[{
        'glm__C': [1e4, 1e5, 1e6],
        'glm__l1_ratio': [0.1, 0.5, 0.9]
    }],
    cv=3
)
grid_search.fit(diabetes['data'], diabetes['target'])
print(grid_search.best_params_)
