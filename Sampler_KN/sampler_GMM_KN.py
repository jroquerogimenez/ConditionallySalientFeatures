from .Utils.GMM_diagonal_optimizers import *

import numpy as np
import scipy as sp
from sklearn import mixture

class GMM_KN(object):

    def __init__(self, n_clustersKN = 1, method_diagonal = 'MI_diagonal_plus', multiKN = 1, **params_kn):
        
        self.n_clustersKN = n_clustersKN
        self.method_diagonal = method_diagonal
        self.multiKN = multiKN  
        self.EM = mixture.GaussianMixture(n_components = self.n_clustersKN)
                
    def fit(self, X):
        self.X = X                                                                         
        self.n_samples, self.dim = self.X.shape[0], self.X.shape[1]            
        self.EM.fit(self.X)      
        self._estimate()            
        
    def sample(self, X_new = None):   
        if X_new is not None:
            assert X_new.shape[1] == self.dim, 'Different input dimensions when fitting and sampling.'
            self._sample(X_new)
        else:
            self._sample(self.X)
        return self.X_KN
        
        
        
    def _estimate(self):
        self.diagonal_clusters, self.SigmaInvDiag_clusters, self.SigmaChol_clusters = {}, {}, {}
        for cluster in np.arange(self.n_clustersKN):
            self.diagonal_clusters[str(cluster)], self.SigmaInvDiag_clusters[str(cluster)], self.SigmaChol_clusters[str(cluster)] = self._KN_distribution(cov_matrix = self.EM.covariances_[cluster,]) 
         
    def _KN_distribution(self, cov_matrix):
        diagonal = create_diagonal(cov_matrix = cov_matrix, multiKN_par = ((self.multiKN + 1)/self.multiKN), method_diagonal = self.method_diagonal)
        SigmaInvDiag = np.linalg.solve(cov_matrix,diagonal)
        
        bool_sampled, eps = False, 1e-8
        while not bool_sampled:
            try:
                SigmaChol = np.linalg.cholesky(  np.tile( (diagonal - diagonal.dot(SigmaInvDiag)), reps = (self.multiKN,self.multiKN)) + np.diag(np.tile(np.diag(diagonal), reps = self.multiKN))+eps*np.identity(self.dim*self.multiKN) )
                bool_sampled = True
            except np.linalg.LinAlgError:
                print('Generated diagonal is invalid. Increasing epsilon x10 for knockoff joint covariance matrix.')
                eps *= 10
                
        return np.diag(diagonal), SigmaInvDiag, SigmaChol    
    
    
    
    
    def _sample(self, X_sample):        
        self.X_KN = np.empty(shape = [X_sample.shape[0], X_sample.shape[1] * self.multiKN])        
        self.assignmentsKN = self._generate_assignments(X_sample)
        for clusterKN in np.arange(self.n_clustersKN):
            self.X_KN[np.where(self.assignmentsKN == clusterKN)[0],:] = self._cluster_samples(X_sample[np.where(self.assignmentsKN == clusterKN)[0],:], clusterKN)
                        
    def _generate_assignments(self, X_sample):
        cat_post = np.array([self.EM.weights_[clusterKN,]*sp.stats.multivariate_normal.pdf(x = X_sample, mean = self.EM.means_[clusterKN,], cov = self.EM.covariances_[clusterKN,]) for clusterKN in np.arange(self.n_clustersKN) ]).T
        cat_post = np.cumsum(cat_post / np.sum(cat_post, axis = 1).reshape(X_sample.shape[0],1), axis = 1)        
        return np.sum(cat_post < np.random.random(size = X_sample.shape[0]).reshape(X_sample.shape[0],1) , axis=1)
        
    def _cluster_samples(self, X_cluster, clusterKN):
        new_mean = np.tile(
            np.matmul(self.EM.means_[clusterKN,].reshape(1,self.dim), self.SigmaInvDiag_clusters[str(clusterKN)])+\
            np.matmul(X_cluster, (np.identity(self.dim) - self.SigmaInvDiag_clusters[str(clusterKN)])),
            reps = self.multiKN)
            
        new_variance = np.matmul(
            self.SigmaChol_clusters[str(clusterKN)], 
            np.random.normal(size = (self.dim*self.multiKN, X_cluster.shape[0]))).T
        
        return new_mean + new_variance
        
