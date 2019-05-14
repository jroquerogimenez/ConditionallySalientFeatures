import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS


class NNClassifierTF():
    
    def __init__(self, dim, Y_train_0, **params_nn):
        self.verbose = params_nn.get('verbose',0)
        self.le = preprocessing.LabelEncoder()    
        self.le.fit(Y_train_0)
        n_classes =  len(np.unique(Y_train_0))        
        tf.reset_default_graph()    
        
        x = tf.placeholder(tf.float32, [None, dim],name='input-x')
        y = tf.placeholder(tf.int32, [None],name='input-y')
        regularizer = tf.contrib.layers.l2_regularizer(params_nn.get('reg_penalty',0.1))
        layer = {'0': tf.layers.dense(inputs = x, units =  FLAGS.num_hidden_nodes, activation = 'relu', kernel_regularizer = regularizer, name = 'dense0')}
        for n_lay in np.arange(1, FLAGS.num_layers):
            layer[str(n_lay)] = tf.add(tf.layers.dense(inputs = layer[str(n_lay - 1)], units =  FLAGS.num_hidden_nodes, activation = 'relu', kernel_regularizer = regularizer, name = 'dense'+str(n_lay)), layer[str(n_lay - 1)], name = 'residual'+str(n_lay))             
        scores = tf.identity(tf.layers.dense(inputs = layer[str(n_lay)], units = n_classes, activation = None, kernel_regularizer = regularizer), name = 'scores')
        print(layer.keys())
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores), name = 'loss')
        gradients = tf.gradients(loss, x, name = 'saliency')
        optimizer = tf.train.AdamOptimizer(name = 'optimizer').minimize(loss)
                
    def fit(self, sess, X_train, Y_train_0, **params_nn):
        Y_train = self.le.transform(Y_train_0)
        batch_size_train = params_nn.get('batch_size_train',1000)
        sess.run(tf.global_variables_initializer())
        for _ in tqdm(np.arange(params_nn.get('n_epochs',20)), desc = 'Epoch training'):
            for _, (x_np, y_np) in enumerate(Dataset(X_train,Y_train,batch_size_train, shuffle = True)):
                sess.run('optimizer', feed_dict = {'input-x:0': x_np, 'input-y:0': y_np})   
        self.saliency_map = None
        if params_nn.get('return_saliency', False):
            saliency_map = np.zeros_like(X_train)
            for i, (x_np, y_np) in enumerate(Dataset(X_train,Y_train,batch_size_train)):
                saliency_np = sess.run('saliency:0', feed_dict={'input-x:0': x_np, 'input-y:0': y_np}) 
                saliency_map[i*batch_size:(i+1)*batch_size,:] = saliency_np[0]
            self.saliency_map = saliency_map
    
    def evaluate(self, sess, X_eval, Y_eval_0, **params_nn):   
        Y_eval = self.le.transform(Y_eval_0)
        num_samples, num_correct = 0,0
        batch_size_test = params_nn.get('batch_size_test',1000)
                
        for i, (x_np, y_np) in enumerate(Dataset(X_eval,Y_eval,batch_size_test)):
            scores_np = sess.run('scores:0', feed_dict={'input-x:0': x_np, 'input-y:0': y_np}) 
            y_pred = scores_np.argmax(axis=1)
            num_samples += x_np.shape[0]
            num_correct += (y_pred == y_np).sum()
            
        return num_correct/num_samples
    
    
    
    
    
    
    
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = np.minimum(batch_size,X.shape[0]), shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B,:], self.y[i:i+B]) for i in range(0, N, B))

