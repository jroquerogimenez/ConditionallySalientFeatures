import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS





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
    plt.savefig(FLAGS.dir_savefig“.pdf”, format=‘pdf’, bbox_inches=‘tight’)
    
    
    
    
    