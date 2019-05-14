from .IntegralSwap import *
from sklearn import linear_model

def IS_procedure(full_covariate, Y, **params_kn):
    method = params_kn.get('method',None)
    saliency_map = None
    if method == 'LogisticRegression':  
        reg = linear_model.LogisticRegression(penalty='l1', solver='saga')
        reg.fit(full_covariate,Y)
        importance_scores = abs(reg.coef_[0,:])
    elif method == 'LinearRegression':
        reg = linear_model.LinearRegression()
        reg.fit(full_covariate,Y)
        importance_scores = abs(reg.coef_)
    elif method == 'LassoCV':
        reg = linear_model.LassoCV()
        reg.fit(full_covariate,Y)
        importance_scores = abs(reg.coef_)
    elif method == 'IntegralSwapLasso':
        importance_scores, saliency_map = IntegralSwapLasso(full_covariate, Y, **params_kn)
    elif method == 'IntegralSwapLassoRegression':
        importance_scores, saliency_map = IntegralSwapLassoRegression(full_covariate, Y, **params_kn)
    else :
        raise ValueError("No other importance score methods are implemented")
        # any other method has to take as input full_covariate (possibly multiKN) and Y and possibly params_kn
        # and return a vector of same dim as full_covariate.shape[1]
    return importance_scores, saliency_map      # returns a 1-d vector