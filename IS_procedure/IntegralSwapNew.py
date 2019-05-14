from Code_KN.Utils.plotting import *
import numpy as np
import tensorflow as tf
from .Networks.resnet import *
tf.logging.set_verbosity(tf.logging.ERROR) 
    
def IntegralSwapLasso(full_covariate, Y, **params_kn):
    '''
    Input: - full_covariate : np.array(n_samples, 2*dim)
           - Y : np.array(n_samples,)
           - params : dict
                      - verbose
                      - classifier : a Class that initiates by training a classifier and has method 'evaluate'
                      - params for classifier
                      - steps_lambda
                      - plot_path : if we want to get the path plotted so that we see if our choice is way off.
    
    Output: - importance_scores : np.array(2*dim,)
    '''
    assert not params_kn.get('isMultiKN',False), 'Multiple knockoffs as input for ScoresSwapLasso'
    dim = int(full_covariate.shape[1]/2)
    
    classif = NNClassifierTF(dim = full_covariate.shape[1], Y_train_0 = Y, **params_kn)
    with tf.Session() as sess:
        #writer = tf.summary.FileWriter('./graphs', sess.graph)
        classif.fit(sess, X_train = full_covariate, Y_train_0 = Y, **params_kn)
        
        initial_accuracy = classif.evaluate(sess, full_covariate, Y, **params_kn)
        print('Baseline accuracy: ', initial_accuracy)
        
        steps_lambda = params_kn.get('steps_lambda', np.arange(1,10,2))
        importance_scores = np.zeros((2*dim, len(steps_lambda)))
        full_covariate_copy = np.copy(full_covariate)
        for index_lambda, cur_lambda in enumerate(steps_lambda):
            if cur_lambda == 0:
                importance_scores[:,index_lambda] = np.zeros(2*dim)
            for covariate in tqdm(np.arange(2*dim), desc = 'Drop at lambda'+str(cur_lambda)):
                full_covariate_copy[:,covariate] = (1-cur_lambda) * full_covariate[:,covariate] + full_covariate[:,(covariate+dim)%(2*dim)]*cur_lambda
                importance_scores[covariate,index_lambda] = initial_accuracy - classif.evaluate(sess, full_covariate_copy,Y, **params_kn)
                full_covariate_copy[:,covariate] = full_covariate[:,covariate]
        #writer.close() 
                
    
    
    integral_importance_scores = np.array([np.trapz(importance_scores[:,:L+1], x = steps_lambda[:L+1], axis = 1) for L in np.arange(len(steps_lambda))]).T
    choice_index_lambda = params_kn.get('choice_index_lambda',len(steps_lambda)-1)        
    if params_kn.get('plot_path', False):
        plot_scores_Lasso(integral_importance_scores, steps_lambda = params_kn.get( 'steps_lambda', np.arange(1,10,2)), non_null = params_kn.get('non_null', None))
            
    return integral_importance_scores[:,choice_index_lambda], classif.saliency_map

