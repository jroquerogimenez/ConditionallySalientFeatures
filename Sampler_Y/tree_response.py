import numpy as np


def logistic(x, intensity):
    dim = x.shape[1]
    logits = intensity*(np.sum(x, axis = 1) - dim)/dim
    probs = 1/(1+np.exp(-logits))
    Y = (np.random.rand(x.shape[0]) < 1/(1+np.exp(-logits)))*1.
    return Y

def linear(x, intensity):
    dim = x.shape[1]
    Y = intensity*(np.sum(x, axis = 1) - dim)
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

def tree_depth_2_logistic(x, cutoff, cutoff2, intensity, **sampler_y_args):
    x_reduced_1 = step_split_reduce(x, cutoff)    
    x_reduced_2 = step_split_reduce(x_reduced_1, cutoff2)
    Y = logistic(x_reduced_2, intensity)
    return Y
    
def tree_depth_1_linear(x, cutoff, intensity, **sampler_y_args):
    x_reduced = step_split_reduce(x, cutoff)
    Y = linear(x_reduced, intensity)
    return Y

def tree_depth_2_linear(x, cutoff, cutoff2, intensity, **sampler_y_args):
    x_reduced_1 = step_split_reduce(x, cutoff)    
    x_reduced_2 = step_split_reduce(x_reduced_1, cutoff2)
    Y = linear(x_reduced_2, intensity)
    return Y    
    
    


