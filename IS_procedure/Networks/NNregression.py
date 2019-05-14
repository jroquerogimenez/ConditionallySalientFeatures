from Code_KN.Utils.plotting import *


import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, plot_model

tf.logging.set_verbosity(tf.logging.ERROR) 


    
def IntegralSwapLassoRegression(full_covariate, Y, **params_kn):
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
    
    classif = NNClassifierTF(**params_kn)
    classif.fit(X_train = full_covariate, Y_train = Y, **params_kn)
    
    initial_loss = classif.evaluate(full_covariate, Y, **params_kn)
    print('Baseline loss: ', initial_loss)
    
    steps_lambda = params_kn.get('steps_lambda', np.arange(1,10,2))
    importance_scores = np.zeros((2*dim, len(steps_lambda)))
    
    for index_lambda, cur_lambda in enumerate(steps_lambda):
        for covariate in np.arange(2*dim):
            if covariate % 100 == 0: print('Computing loss increase for covariate', covariate, 'at lambda: ',cur_lambda)
            full_covariate_swap_local = np.copy(full_covariate)
            full_covariate_swap_local[:,covariate] = (full_covariate[:,(covariate+dim)%(2*dim)] - full_covariate[:,covariate])*cur_lambda + full_covariate[:,covariate]
            importance_scores[covariate,index_lambda] = classif.evaluate(full_covariate_swap_local,Y, **params_kn) - initial_loss
    
    integral_importance_scores = np.array([np.trapz(importance_scores[:,:L+1], x = steps_lambda[:L+1], axis = 1) for L in np.arange(len(steps_lambda))]).T
    
    
    
    saliency_map = None
    if params_kn.get('return_saliency', False):
        saliency_map = classif.saliency(X_eval = full_covariate, Y_eval = Y, **params_kn)
        
        
    if params_kn.get('plot_path', False):
        plot_scores_Lasso(integral_importance_scores, steps_lambda = steps_lambda, non_null = params_kn.get('non_null', None))
    
    choice_index_lambda = params_kn.get('choice_index_lambda',len(steps_lambda)//2)    
    return integral_importance_scores[:,choice_index_lambda], saliency_map





class NNRegressorTF():
    
    def __init__(self, **params_nn):
        self.verbose = params_nn.get('verbose',0)

    def fit(self, X_train, Y_train, **params_nn):
        self.dim_output = Y_train.shape[1] if len(Y_train.shape) > 1 else 1
        self.dim_input = X_train.shape[1]
        self.saver = train_graph(X_train, Y_train, dim_input = self.dim_input, dim_output = self.dim_output, **params_nn)
        
    def evaluate(self, X_eval, Y_eval, **params_nn):
        return evaluate_graph(X_eval, Y_eval, self.saver, dim_input = self.dim_input, dim_output = self.dim_output, **params_nn)
    
    def saliency(self, X_eval, Y_eval, **params_nn):
        return saliency_map(X_eval, Y_eval, self.saver, dim_input = self.dim_input, dim_output = self.dim_output, **params_nn)
    
def layer_fc(x, params):
    w1, w2, w3, b1, b2, b3 = params  
    h1 = tf.nn.relu(tf.matmul(x, w1)+b1, name= 'hiden-layer-1') 
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2, name= 'hiden-layer-2')
    output = tf.matmul(h2, w3)+b3        
    return output

def layer_fc_init(dim_input, dim_output, **params_nn):
    h_dim = params_nn.get('hidden_dim', 300)
    w1 = tf.Variable(tf.random_normal(shape = (dim_input, h_dim)), name='w1')
    w2 = tf.Variable(tf.random_normal(shape = (h_dim, h_dim)), name='w2')
    w3 = tf.Variable(tf.random_normal(shape = (h_dim, dim_output)), name='w3')
    b1 = tf.Variable(tf.zeros((h_dim), name= 'b1-init'), name='b1')
    b2 = tf.Variable(tf.zeros((h_dim), name= 'b2-init'), name='b2')
    b3 = tf.Variable(tf.zeros((dim_output), name= 'b3-init'), name='b3')
    return [w1, w2, w3, b1, b2, b3]

def train_graph(X_train, Y_train, dim_input, dim_output,**params_nn):    
    tf.reset_default_graph()    
    
    x = tf.placeholder(tf.float32, [None, None],name='input-x')
    y = tf.placeholder(tf.float32, [None],name='input-y')
    params = layer_fc_init(dim_input, dim_output, **params_nn)          
    output = tf.reshape(layer_fc(x, params), [-1] )    
    loss = tf.nn.l2_loss(output-y)
    loss += params_nn.get('reg_penalty',1)*tf.reduce_sum([tf.nn.l2_loss(param) for param in params])
    
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in np.arange(params_nn.get('n_epochs',20)):
            for _, (x_np, y_np) in enumerate(Dataset(X_train,Y_train,params_nn.get('batch_size',1000), shuffle = False)):
                print('Training loss: ', sess.run(loss, feed_dict = {x: x_np, y: y_np}))
                _ = sess.run(optimizer, feed_dict = {x: x_np, y: y_np})
        saver.save(sess,params_nn.get('sess_store_filename','./simu.ckpt'))
    return saver           
                

def evaluate_graph(X_eval, Y_eval, saver, dim_input, dim_output, **params_nn):    
    num_samples, total_loss = 0,0
    tf.reset_default_graph()    
    
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None])
    params = layer_fc_init(dim_input, dim_output, **params_nn)         
    output = tf.reshape(layer_fc(x, params), [-1] )
    loss = tf.nn.l2_loss(output-y)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,params_nn.get('sess_store_filename','./simu.ckpt'))
        for _, (x_np, y_np) in enumerate(Dataset(X_eval, Y_eval,params_nn.get('batch_size',1000), shuffle = False)):
            print('output', sess.run(output, feed_dict={x: x_np, y: y_np}))
            print('y_np', y_np)
            print('loss', sess.run(loss, feed_dict={x: x_np, y: y_np}) )
            total_loss += sess.run(loss, feed_dict={x: x_np, y: y_np}) 
            num_samples += x_np.shape[0]
    return total_loss/num_samples
    
    
def saliency_map(X_eval, Y_eval, saver, dim_input, dim_output, **params_nn):    
    tf.reset_default_graph()    
    
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None])
    params = layer_fc_init(dim_input, dim_output, **params_nn)         
    output = tf.reshape(layer_fc(x, params), [-1] )
    loss = tf.nn.l2_loss(y-output)
    
    gradients = tf.gradients(loss, x)
    saliency_map = np.zeros_like(X_eval)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,params_nn.get('sess_store_filename','./simu.ckpt'))
        for i, (x_np, y_np) in enumerate(Dataset(X_eval, Y_eval,params_nn.get('batch_size',1000), shuffle = False)):
            saliency_batch = sess.run(gradients, feed_dict={x: x_np, y: y_np})
            batch_size = params_nn.get('batch_size',1000)
            saliency_map[i*batch_size:(i+1)*batch_size,:] = saliency_batch[0]
        return saliency_map
    
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B,:], self.y[i:i+B]) for i in range(0, N, B))
