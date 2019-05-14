from sklearn import linear_model
import numpy as np

def LogisticLassoSelection(X, Y , K):
    reg = linear_model.LogisticRegression()
    X_transf = X/X.shape[1]*4 - 1
    reg.fit(X_transf,Y)
    print(X[:3,:])
    print(Y[:3])
    print(abs(reg.coef_[0,:]))
    return np.sort(np.argsort(abs(reg.coef_[0,:]))[:K])




def logistic(x, intensity):
    dim = x.shape[1]
    logits = intensity*(np.sum(x, axis = 1) - dim)/dim
    probs = 1/(1+np.exp(-logits))
    Y = (np.random.rand(x.shape[0]) < 1/(1+np.exp(-logits)))*1.
    return Y


def step_split_reduce(x, cutoff):
    dim = x.shape[1]
    n_cols = int((dim-1)/2)
    x_reduced = np.zeros((x.shape[0], n_cols))
    x_reduced[np.where(x[:,0] >= cutoff)[0],:] = x[np.where(x[:,0] >= cutoff)[0],1:(n_cols + 1)]
    x_reduced[np.where(x[:,0] < cutoff)[0],:] = x[np.where(x[:,0] < cutoff)[0],(n_cols + 1):2*n_cols+1]
    return x_reduced

def tree_depth_1_logistic(x, cutoff, intensity, **sampler_y_args):
    x_reduced = step_split_reduce(x, cutoff)
    Y = logistic(x_reduced, intensity)
    return Y