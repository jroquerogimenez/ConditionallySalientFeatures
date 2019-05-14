import numpy as np
import matplotlib.pyplot as plt


def plot_scores(importance_scores, W, threshold):
    plt.plot(np.arange(len(importance_scores)),importance_scores, 'r+')
    plt.plot(np.arange(len(W)),W, 'g^')
    if threshold is not None:
        plt.axhline(y=threshold)
    plt.savefig('fig'+'.pdf', format='pdf', bbox_inches='tight')
    #plt.show()
    
        
def plot_scores_Lasso(importance_scores, steps_lambda, non_null=None):
    dim, n_steps = int(importance_scores.shape[0]/2), importance_scores.shape[1]
    W = importance_scores[:dim,:] - importance_scores[dim:,:]
    
    for ind in np.arange(dim):
        if non_null is not None:
            if np.in1d([ind], non_null):
                plt.plot(steps_lambda, W[ind,:], c='r')
            else:
                plt.plot(steps_lambda, W[ind,:], c='g')
        else : 
            plt.plot(steps_lambda, W[ind,:]) 
    plt.show()


def plot_index(index, values):
    '''
    When bootstrap, plot for a given index the set of T values for that index when it was randomly selected among the d(kappa + 1)
    values is a dict with keys = str(i) for i in the index set
    '''
    _ = plt.hist(values[str(index)],bins=400)
    _ = plt.title('Index '+str(index))
    plt.show()
    
    
