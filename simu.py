import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, itertools, csv
import scipy as sp
from tqdm import tqdm
import Code_KN
import tensorflow as tf
dictionary_save = {}

FLAGS = tf.app.flags.FLAGS

##########################################
# Good examples:
# 
# python simu.py --num_dim=500 --num_samples=50000 --num_non_null=50 --sampler_y='tree_depth_2_logistic' --cutoff=1.5

# python simu.py --num_dim=100 --num_samples=10000 --num_non_null=30 --sampler_x='GMM' --sampler_kn='GMM' --sampler_y='tree_depth_1_linear' --is_method='LinearRegression' --cutoff=0

# python simu.py --num_dim=100 --num_samples=10000 --num_non_null=30 --sampler_x='GMM' --sampler_kn='GMM' --sampler_y='tree_depth_2_linear' --is_method='LinearRegression' --cutoff=0

# python simu.py --num_dim=60 --num_samples=10000 --num_non_null=20 --is_method='IntegralSwapLasso' --cutoff=1.5
# 
# 
##########################################
tf.app.flags.DEFINE_string('sampler_x', 'HMM', "")

tf.app.flags.DEFINE_string('sampler_kn', 'HMM', "")

tf.app.flags.DEFINE_string('sampler_y', 'tree_depth_1_logistic', "")

tf.app.flags.DEFINE_integer('num_dim', 100, "Number of dimensions.")

tf.app.flags.DEFINE_integer('num_samples', 10000, "Number of samples.")

tf.app.flags.DEFINE_integer('num_non_null', 7, "Number of non-null features.")

tf.app.flags.DEFINE_float('FDR', 0.2, "Target FDR.")

tf.app.flags.DEFINE_float('offset', 1, "Offset for knockoff/knockoff+ procedure.")

tf.app.flags.DEFINE_integer('num_clusters', 3, "Number of hidden clusters for GMM/TMM generation.")

tf.app.flags.DEFINE_integer('num_clusters_kn', 3, "Number of hidden clusters for GMM KN.")

tf.app.flags.DEFINE_integer('num_hidden_states', 3, "Number of hidden states for HMM generation.")

tf.app.flags.DEFINE_integer('num_hidden_states_kn', 3, "Number of hidden states for KN HMM.")

tf.app.flags.DEFINE_integer('num_emission_states', 3, "Number of emission states for HMM.")

tf.app.flags.DEFINE_integer('num_populations', 3, "Number of populations in ADM generation.")

tf.app.flags.DEFINE_integer('num_populations_kn', 3, "Number of populations for KN ADM.")

##########################################
tf.app.flags.DEFINE_string('is_method', 'LogisticRegression', "Method to compute importance scores.")

tf.app.flags.DEFINE_string('architecture', 'resnet', "Network model used to compute importance scores.")

tf.app.flags.DEFINE_integer('num_layers', 8, "Number of layers of the network.")

tf.app.flags.DEFINE_integer('num_hidden_nodes', 64, "Number of hidden dimensions of each layer.")

tf.app.flags.DEFINE_integer('num_epochs', 50, "Number of epochs during training process.")

tf.app.flags.DEFINE_integer('batch_train', 1000, "Batch size during training.")

tf.app.flags.DEFINE_integer('batch_eval', 1000, "Batch size during evaluation.")

tf.app.flags.DEFINE_float('regularization_pen', 10., "Regularization strength for the coefficient penalty.")


tf.app.flags.DEFINE_integer('search_partition', 0, "Automatic split of the feature space")

tf.app.flags.DEFINE_bool('plotting_scores', False, "Output a visualization of the importance scores.")

tf.app.flags.DEFINE_bool('plotting_path', False, "Output a visualization of the lambda path.")

tf.app.flags.DEFINE_bool('return_saliency', False, "Return the saliency scores of the dataset.")

tf.app.flags.DEFINE_string('dir_saveval', None, "Directory name to save values.")

tf.app.flags.DEFINE_string('dir_savefig', None, "Directory name to save figures")

tf.app.flags.DEFINE_bool('write_result', False, "Write the output to csv.")
####################################################################
# Parameters for the response generating process are not yet given by flags.
tf.app.flags.DEFINE_float('cutoff', 1.5, 'Tree partition cutoff.')
tf.app.flags.DEFINE_float('intensity', 10., 'Signal intensity.')

intensity = FLAGS.intensity
cutoff = FLAGS.cutoff
####################################################################

Sampler_X_class = getattr(Code_KN.Sampler_X, FLAGS.sampler_x+'_X')
Sampler_KN_class = getattr(Code_KN.Sampler_KN, FLAGS.sampler_kn+'_KN')
Sampler_Y_function = getattr(Code_KN.Sampler_Y, FLAGS.sampler_y)

sampler_x_args = {'num_clusters': FLAGS.num_clusters, 'num_hidden_states': FLAGS.num_hidden_states, 'num_populations': FLAGS.num_populations, 'num_emission_states': FLAGS.num_emission_states}
sampler_kn_args = {'num_clusters_kn': FLAGS.num_clusters_kn, 'num_hidden_states_kn' : FLAGS.num_hidden_states_kn, 'num_populations_kn': FLAGS.num_populations_kn}
sampler_y_args = {'cutoff' : cutoff, 'cutoff2':cutoff, 'intensity' : intensity}

print('\n')
print('--------------------------------------------------------------')
print('Generating synthetic data')
print('\n')
Sampler_X = Sampler_X_class(num_samples = FLAGS.num_samples, num_dim = FLAGS.num_dim, **sampler_x_args)
Sampler_KN = Sampler_KN_class(**sampler_kn_args)
Sampler_KN.fit(Sampler_X.X)
_ = Sampler_KN.sample()
X = Sampler_X.X
X_KN = Sampler_KN.X_KN

non_null = np.sort(np.random.choice(np.arange(FLAGS.num_dim), size = FLAGS.num_non_null, replace = False))
Y = Sampler_Y_function(x = X[:,non_null], **sampler_y_args)

####################################################################

params_kn = {
             'FDR':FLAGS.FDR,                          
             'offset':FLAGS.offset,                 
             'method':FLAGS.is_method,             
             'batch_size_train': FLAGS.batch_train,
             'batch_size_test': FLAGS.batch_eval,
             'n_epochs': FLAGS.num_epochs,
             'steps_lambda': np.arange(0,3,0.2),
             'reg_penalty': FLAGS.regularization_pen,
             'hidden_dim' : FLAGS.num_hidden_nodes,
             'n_layers': FLAGS.num_layers,
             'plotting_scores': FLAGS.plotting_scores,
             'plotting_path' : FLAGS.plotting_path,
             'return_saliency': FLAGS.return_saliency,
            }

print('\n')
print('--------------------------------------------------------------')
print('Parameters: ', params_kn, )

knockoff_procedure = Code_KN.KN_procedure.knockoff_procedure
knockoff_performance = Code_KN.KN_procedure.knockoff_performance

knock = knockoff_procedure(**params_kn)
knock.get_importance_scores(X, X_KN, Y, **params_kn)
knock.get_selections(**params_kn)
FDR, Power = knockoff_performance(selected = knock.selected, true = non_null)

print('\n')
print('Selected features on whole data: ', knock.selected)
print('Global non-nulls: ', non_null)
print('FDR: ', FDR)
print('Power: ', Power)

####################################################################
# Running the knockoff procedure on local subregions.

pivot_0, half_dim  = non_null[0], int((FLAGS.num_non_null-1)/2)
quart_dim = int((half_dim-1)/2)
pivot_00 = non_null[1]
pivot_01 = non_null[half_dim +1]
if FLAGS.search_partition == 0:
    # We assume an oracle tells us which are the subregions.    
    print('\n')
    print('--------------------------------------------------------------')
    print('FIRST-LEVEL SPLIT')
    
    X_0, X_1, X_KN_0, X_KN_1, Y_0, Y_1 = Code_KN.KN_procedure.split_partition_data(X, X_KN, Y, pivot_0, cutoff)
    knock_0 = knockoff_procedure(**params_kn)
    knock_0.get_importance_scores(X_0, X_KN_0, Y_0,**params_kn)
    knock_0.get_selections(**params_kn)
    knock_1 = knockoff_procedure(**params_kn)
    knock_1.get_importance_scores(X_1, X_KN_1, Y_1,**params_kn)
    knock_1.get_selections(**params_kn)
    
    print('\n')
    print('Selected features on first partition in first-level split: ', knock_0.selected)
    print('Non-nulls on the first partition in first-level split: ', non_null[1:half_dim + 1])
    print('Number of samples: ', len(Y_0))
    FDR_0, Power_0 = knockoff_performance(knock_0.selected, non_null[1:half_dim + 1])
    print('FDR: ', FDR_0)
    print('Power: ', Power_0)
    print('\n')
    print('Selected features on second partition in first-level split: ', knock_1.selected)
    print('Non-nulls on the second partition in first-level split: ', non_null[half_dim + 1:2*half_dim + 1])
    print('Number of samples: ', len(Y_1))
    FDR_1, Power_1 = knockoff_performance(knock_1.selected, non_null[half_dim + 1:2*half_dim + 1])
    print('FDR: ', FDR_1)
    print('Power: ', Power_1)
    
     
    
    print('\n')
    print('--------------------------------------------------------------')
    print('SECOND-LEVEL SPLIT') 
    
    X_00, X_01, X_KN_00, X_KN_01, Y_00, Y_01 = Code_KN.KN_procedure.split_partition_data(X_0, X_KN_0, Y_0, pivot_00, cutoff)
    knock_00 = knockoff_procedure(**params_kn)
    knock_00.get_importance_scores(X_00, X_KN_00, Y_00,**params_kn)
    knock_00.get_selections(**params_kn)
    knock_01 = knockoff_procedure(**params_kn)
    knock_01.get_importance_scores(X_01, X_KN_01, Y_01,**params_kn)
    knock_01.get_selections(**params_kn)
    
    X_10, X_11, X_KN_10, X_KN_11, Y_10, Y_11 = Code_KN.KN_procedure.split_partition_data(X_1, X_KN_1, Y_1, pivot_01, cutoff)
    knock_10 = knockoff_procedure(**params_kn)
    knock_10.get_importance_scores(X_10, X_KN_10, Y_10,**params_kn)
    knock_10.get_selections(**params_kn)
    knock_11 = knockoff_procedure(**params_kn)
    knock_11.get_importance_scores(X_11, X_KN_11, Y_11,**params_kn)
    knock_11.get_selections(**params_kn)
    
    print('\n')
    print('Selected features on first partition in second-level split: ', knock_00.selected)
    print('Non-nulls on the first partition in second-level split: ', non_null[2:quart_dim + 2])
    print('Number of samples: ', len(Y_00))
    FDR_00, Power_00 = knockoff_performance(knock_00.selected, non_null[2:quart_dim + 2])
    print('FDR: ', FDR_00)
    print('Power: ', Power_00)
    print('\n')
    print('Selected features on second partition in second-level split: ', knock_01.selected)
    print('Non-nulls on the second partition in second-level split: ', non_null[quart_dim + 2:2*quart_dim + 2])
    print('Number of samples: ', len(Y_01))
    FDR_01, Power_01 = knockoff_performance(knock_01.selected, non_null[quart_dim + 2:2*quart_dim + 2])
    print('FDR: ', FDR_01)
    print('Power: ', Power_01)
    
    
    print('\n')
    print('Selected features on third partition in second-level split: ', knock_10.selected)
    print('Non-nulls on the third partition in second-level split: ', non_null[half_dim + 2:half_dim + quart_dim + 2])
    print('Number of samples: ', len(Y_10))
    FDR_10, Power_10 = knockoff_performance(knock_10.selected, non_null[half_dim + 2:half_dim + quart_dim + 2])
    print('FDR: ', FDR_10)
    print('Power: ', Power_10)
    print('\n')
    print('Selected features on fourth partition in second-level split: ', knock_11.selected)
    print('Non-nulls on the fourth partition in second-level split: ', non_null[half_dim + quart_dim + 2:half_dim + 2*quart_dim + 2])
    print('Number of samples: ', len(Y_11))
    FDR_11, Power_11 = knockoff_performance(knock_11.selected, non_null[half_dim + quart_dim + 2:half_dim + 2*quart_dim + 2])
    print('FDR: ', FDR_11)
    print('Power: ', Power_11)
    
    print('\n')
    print('Selection through Logistic regression with lasso')
    print('\n')
    print('--------------------------------------------')
    print('Whole data')
    selected_LR = Code_KN.KN_procedure.LogisticLassoSelection(X,Y, K=20)
    print('Selected features on the whole data: ', selected_LR)
    print('Performance: ')
    FDR_LR, Power_LR = knockoff_performance(selected_LR, non_null)
    print('FDR: ', FDR_LR)
    print('Power: ', Power_LR)
    print('\n')
    print('--------------------------------------------')
    print('First partition of first split')
    selected_LR_0 = Code_KN.KN_procedure.LogisticLassoSelection(X_0,Y_0, K=20)
    print('Selected features: ', selected_LR_0)
    print('Performance: ')
    FDR_LR_0, Power_LR_0 = knockoff_performance(selected_LR_0, non_null[1:half_dim + 1])
    print('FDR: ', FDR_LR_0)
    print('Power: ', Power_LR_0)
    print('\n')
    print('Second partition of first split')
    selected_LR_1 = Code_KN.KN_procedure.LogisticLassoSelection(X_1,Y_1, K=20)
    print('Selected features: ', selected_LR_1)
    print('Performance: ')
    FDR_LR_1, Power_LR_1 = knockoff_performance(selected_LR_1, non_null[half_dim + 1:2*half_dim + 1])
    print('FDR: ', FDR_LR_1)
    print('Power: ', Power_LR_1)
    print('\n')
    print('--------------------------------------------')
    print('First partition of second split')
    selected_LR_00 = Code_KN.KN_procedure.LogisticLassoSelection(X_00,Y_00, K=20)
    print('Selected features: ', selected_LR_00)
    print('Performance: ')
    FDR_LR_00, Power_LR_00 = knockoff_performance(selected_LR_00, non_null[2:quart_dim + 2])
    print('FDR: ', FDR_LR_00)
    print('Power: ', Power_LR_00)
    print('\n')
    print('Second partition of second split')
    selected_LR_01 = Code_KN.KN_procedure.LogisticLassoSelection(X_01,Y_01, K=20)
    print('Selected features: ', selected_LR_01)
    print('Performance: ')
    FDR_LR_01, Power_LR_01 = knockoff_performance(selected_LR_01, non_null[quart_dim + 2:2*quart_dim + 2])
    print('FDR: ', FDR_LR_01)
    print('Power: ', Power_LR_01)
    print('\n')
    print('Third partition of second split')
    selected_LR_10 = Code_KN.KN_procedure.LogisticLassoSelection(X_10,Y_10, K=20)
    print('Selected features: ', selected_LR_10)
    print('Performance: ')
    FDR_LR_10, Power_LR_10 = knockoff_performance(selected_LR_10, non_null[half_dim + 2:half_dim + quart_dim + 2])
    print('FDR: ', FDR_LR_10)
    print('Power: ', Power_LR_10)
    print('\n')
    print('Fourth partition of second split')
    selected_LR_11 = Code_KN.KN_procedure.LogisticLassoSelection(X_11,Y_11, K=20)
    print('Selected features: ', selected_LR_11)
    print('Performance: ')
    FDR_LR_11, Power_LR_11 = knockoff_performance(selected_LR_11, non_null[half_dim + quart_dim + 2:half_dim + 2*quart_dim + 2])
    print('FDR: ', FDR_LR_11)
    print('Power: ', Power_LR_11)
    print('\n')
    
    FDR_lev_0   = FDR
    Power_lev_0 = Power
    FDR_lev_1   = np.mean([FDR_0, FDR_1])
    FDR_lev_2   = np.mean([FDR_00, FDR_01, FDR_10, FDR_11])
    Power_lev_1 = np.mean([Power_0, Power_1])
    Power_lev_2 = np.mean([Power_00, Power_01, Power_10, Power_11])
    
    fdr_1,_ = knockoff_performance(knock.selected,non_null[2:quart_dim + 2])
    fdr_2,_ = knockoff_performance(knock.selected,non_null[quart_dim + 2:2*quart_dim + 2])
    fdr_3,_ = knockoff_performance(knock.selected,non_null[half_dim + 2:half_dim + quart_dim + 2])
    fdr_4,_ = knockoff_performance(knock.selected,non_null[half_dim + quart_dim + 2:half_dim + 2*quart_dim + 2])
    
    FDR_extra_02 = np.mean([fdr_1,fdr_2,fdr_3,fdr_4])
    
    fdr_1,_ = knockoff_performance(knock_0.selected,non_null[2:quart_dim + 2])
    fdr_2,_ = knockoff_performance(knock_0.selected,non_null[quart_dim + 2:2*quart_dim + 2])
    fdr_3,_ = knockoff_performance(knock_1.selected,non_null[half_dim + 2:half_dim + quart_dim + 2])
    fdr_4,_ = knockoff_performance(knock_1.selected,non_null[half_dim + quart_dim + 2:half_dim + 2*quart_dim + 2])
    
    FDR_extra_12 = np.mean([fdr_1,fdr_2,fdr_3,fdr_4])
    
    line_condensed = [str(FDR_lev_0),str(Power_lev_0),str(FDR_lev_1),str(FDR_lev_2),str(Power_lev_1),str(Power_lev_2),str(FDR_extra_02),str(FDR_extra_12)]

    
    
    
    if FLAGS.write_result:
        with open('results.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow([str(FDR), str(Power), str(FDR_0), str(Power_0), str(FDR_1), str(Power_1), str(FDR_00), str(Power_00), str(FDR_01), str(Power_01), str(FDR_10), str(Power_10), str(FDR_11), str(Power_11)])
    
    if FLAGS.write_result:
        with open('results_condensed.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(line_condensed)
    
    
elif FLAGS.search_partition == 1: 
    # We assume that we search a partition of the feature space into two sub-regions.
    print('Search for a one-level partition')
    new_data = X[:,knock.selected]    
    boundaries = [0.5,1.5]
    drop_array = np.zeros((len(knock.selected), len(boundaries), len(knock.selected), 2))
    NNClassifierTF = Code_KN.IS_procedure.Networks.resnet.NNClassifierTF
    classifier_partition_full = NNClassifierTF(dim = len(knock.selected), Y_train_0 = Y, **params_kn)
    with tf.Session() as sess:
        for i, (index, boundary) in tqdm(itertools.product(np.arange(len(knock.selected)), enumerate(boundaries)), desc = 'Eval at partition', total = len(boundaries)*len(knock.selected)):
            partition_0 = new_data[np.where(new_data[:,i] > boundary)[0],:]
            Y_0 = Y[np.where(new_data[:,i] > boundary)[0]]
            classifier_partition_full.fit(sess = sess, X_train = partition_0, Y_train_0 = Y_0, **params_kn)
            drop_array[i, index, :, 0] = classifier_partition_full.evaluate(sess, partition_0, Y_0, **params_kn)
            
            partition_1 = new_data[np.where(new_data[:,i] < boundary)[0],:]
            Y_1 = Y[np.where(new_data[:,i] < boundary)[0]]
            classifier_partition_full.fit(sess, partition_1, Y_1, **params_kn)
            drop_array[i, index, :, 1] =  classifier_partition_full.evaluate(sess, partition_1, Y_1, **params_kn)
    
    
    classifier_partition_red = NNClassifierTF(dim = len(knock.selected)-1, Y_train_0 = Y, **params_kn)
    with tf.Session() as sess:
        for i, (index, boundary) in tqdm(itertools.product(np.arange(len(knock.selected)), enumerate(boundaries)), desc = 'Eval at partition', total = len(boundaries)*len(knock.selected)):
            partition_0 = new_data[np.where(new_data[:,i] > boundary)[0],:]
            Y_0 = Y[np.where(new_data[:,i] > boundary)[0]]
            partition_1 = new_data[np.where(new_data[:,i] < boundary)[0],:]
            Y_1 = Y[np.where(new_data[:,i] < boundary)[0]]
            for j in np.arange(len(knock.selected)):
                    data_dropped_0 = np.delete(partition_0, j, axis = 1)
                    classifier_partition_red.fit(sess = sess, X_train = data_dropped_0, Y_train_0 = Y_0, **params_kn)
                    drop_array[i, index, j, 0] -= classifier_partition_red.evaluate(sess, data_dropped_0, Y_0, **params_kn)
    
                    data_dropped_1 = np.delete(partition_1, j, axis = 1)                    
                    classifier_partition_red.fit(sess, data_dropped_1, Y_1, **params_kn)
                    drop_array[i, index, j, 1] -= classifier_partition_red.evaluate(sess, data_dropped_1, Y_1, **params_kn)
        
    print(drop_array)
    drop_array = np.maximum(drop_array,0)    
    index_pivot, index_boundary_pivot, objective_pivot = np.unravel_index(np.argmax(np.abs(drop_array[:,:,:,0] - drop_array[:,:,:,1])), drop_array.shape[:-1])    
    pivot = knock.selected[index_pivot]
    boundary_pivot = boundaries[index_boundary_pivot]
    print('Pivot value', pivot,'Boundary value', boundary_pivot)
    partition_0_X = X[np.where(X[:,pivot] > boundary_pivot)[0],:]
    partition_0_KN = X_KN[np.where(X[:,pivot] > boundary_pivot)[0],:]
    partition_0_Y = Y[np.where(X[:,pivot] > boundary_pivot)[0]]
    print(len(partition_0_Y))
    partition_1_X = X[np.where(X[:,pivot] < boundary_pivot)[0],:]
    partition_1_KN = X_KN[np.where(X[:,pivot] < boundary_pivot)[0],:]
    partition_1_Y = Y[np.where(X[:,pivot] < boundary_pivot)[0]]
    print(len(partition_1_Y))
        
        
    knock_partition_0 = knockoff_procedure(offset = params_kn['offset'])
    knock_partition_0.get_importance_scores(partition_0_X, partition_0_KN, partition_0_Y, **params_kn)
    knock_partition_0.get_selections(**params_kn)
    knockoff_performance(selected = knock_partition_0.selected, true = non_null)
    print(knock_partition_0.selected, non_null)    
        
        
    knock_partition_1 = knockoff_procedure(offset = params_kn['offset'])
    knock_partition_1.get_importance_scores(partition_1_X, partition_1_KN, partition_1_Y, **params_kn)
    knock_partition_1.get_selections(**params_kn)
    knockoff_performance(selected = knock_partition_1.selected, true = non_null)
    print(knock_partition_1.selected, non_null)    
        
        
        
    print('Oracle partition of the space.')
    
    pivot = non_null[0]
    boundary_pivot = 1.5
    print('Pivot value', pivot,'Boundary value', boundary_pivot)
    partition_0_X = X[np.where(X[:,pivot] > boundary_pivot)[0],:]
    partition_0_KN = X_KN[np.where(X[:,pivot] > boundary_pivot)[0],:]
    partition_0_Y = Y[np.where(X[:,pivot] > boundary_pivot)[0]]
    print(len(partition_0_Y))
    partition_1_X = X[np.where(X[:,pivot] < boundary_pivot)[0],:]
    partition_1_KN = X_KN[np.where(X[:,pivot] < boundary_pivot)[0],:]
    partition_1_Y = Y[np.where(X[:,pivot] < boundary_pivot)[0]]
    print(len(partition_1_Y))
        
        
    knock_partition_0 = knockoff_procedure(offset = params_kn['offset'])
    knock_partition_0.get_importance_scores(partition_0_X, partition_0_KN, partition_0_Y, **params_kn)
    knock_partition_0.get_selections(**params_kn)
    knockoff_performance(selected = knock_partition_0.selected, true = non_null)
    print(knock_partition_0.selected, non_null)    
        
        
    knock_partition_1 = knockoff_procedure(offset = params_kn['offset'])
    knock_partition_1.get_importance_scores(partition_1_X, partition_1_KN, partition_1_Y, **params_kn)
    knock_partition_1.get_selections(**params_kn)
    knockoff_performance(selected = knock_partition_1.selected, true = non_null)
    print(knock_partition_1.selected, non_null)    
    
    


