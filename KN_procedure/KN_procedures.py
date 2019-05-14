from Code_KN.Utils.plotting import *
from Code_KN.IS_procedure.IS_procedures import *
import numpy as np


class knockoff_procedure(object):

    def __init__(self, FDR=0.2, offset = 1, **params_kn):
        self.FDR = FDR
        self.offset = offset
        self.importance_scores = None
        self.selected = None
        self.saliency_map = None
    def get_importance_scores(self, X, X_KN, Y, **params_kn):
        '''
          Updates:
        self.importance_scores -- type np.array(dim*something,1): one-dimensional vector
        '''
        if (params_kn.get('n_bootstr',0) > 0)&(int(X_KN.shape[1]/X.shape[1]) > 1):
            self.importance_scores = bootstraps(X = X, X_KN = X_KN, Y = Y, **params_kn)   # returns a 1-d vector
        else:
            self.importance_scores, self.saliency_map = IS_procedure(full_covariate = np.hstack([X,X_KN]), Y = Y, **params_kn)
        self.dim, self.multiKN = X.shape[1], int(X_KN.shape[1]/X.shape[1])
        
            
    def get_selections(self, **params_kn):
        '''
          Needs:
        self.importance_scores not None 
          Updates:
        self.selected
        '''
        W = np.array(list(map(lambda x,y : x-y, self.importance_scores[0:self.dim], self.importance_scores[self.dim: 2*self.dim])))
        sorted_abs_W, ratios = np.sort(np.absolute(W)), []
        for index in np.arange(self.dim):
            above = np.count_nonzero([x >= sorted_abs_W[index] for x in W])
            below = np.count_nonzero([x <= - sorted_abs_W[index] for x in W])
            ratios.append(((self.offset + below)  /  np.maximum(above,0.001)))
        if np.sum([ratio<self.FDR for ratio in ratios])==0 :
            threshold, self.selected = None, []
        else:
            threshold = sorted_abs_W[np.min([ind for ind in np.arange(self.dim) if ratios[ind]<self.FDR ])]
            self.selected = np.where(W >= threshold)[0]
        self.W = W    
        if params_kn.get('plotting_scores', False):
            plot_scores(self.importance_scores, self.W, threshold)
            
        
    def get_selections_multikn(self, **params_kn):  # UPDATE: THRESHOLD FOR RATIOS
        '''
          Needs:
        self.importance_scores not None 
          Updates:
        self.selected
        '''
        assert self.multiKN > 1, 'Need multiple knockoffs.'
        importance_scores_reshaped = self.importance_scores.reshape((self.multiKN + 1), self.dim)
        i_max, t_max = np.argmax(importance_scores_reshaped, axis = 0), np.amax(importance_scores_reshaped, axis = 0)
        taus = np.sort(t_max - importance_scores_reshaped, axis = 0)[1,:]
        ratios = np.array( [((np.sum((taus > tau)&(i_max > 0)))+self.offset)/np.maximum(np.sum((taus > tau)&(i_max == 0)),1) < self.FDR*self.multiKN for tau in taus] )
        if np.sum(ratios) == 0:
            threshold, self.selected = None, []
        else:
            threshold = np.amin(taus[np.nonzero(ratios)])
            self.selected = np.nonzero((taus > threshold)&(i_max == 0))[0]
                
        if params_kn.get('plotting_scores', False):
            plot_scores(self.importance_scores, taus, threshold)



            
def knockoff_performance(selected, true):
    if len(selected) == 0:
        return 0,0
    else:
        FDR = len(set(selected).difference(set(true)))/len(selected)
        Power = len(set(selected).intersection(set(true)))/len(true)        
        return FDR, Power

def split_partition_data(X, X_KN, Y, pivot, cutoff): 
    selected_indices_0 = list(set(np.where(X[:,pivot] > cutoff)[0]).intersection(set(np.where(X_KN[:,pivot] > cutoff)[0])))
    selected_indices_1 = list(set(np.where(X[:,pivot] < cutoff)[0]).intersection(set(np.where(X_KN[:,pivot] < cutoff)[0])))
    X_0, X_KN_0, Y_0 = X[selected_indices_0,:],X_KN[selected_indices_0,:],Y[selected_indices_0]
    X_1, X_KN_1, Y_1 = X[selected_indices_1,:],X_KN[selected_indices_1,:],Y[selected_indices_1]
    return X_0, X_1, X_KN_0, X_KN_1, Y_0, Y_1
    
    