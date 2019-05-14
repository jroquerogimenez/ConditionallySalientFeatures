import numpy as np
from sklearn import datasets

def normalize(v):
    if np.sum(v) == 0:
        w=np.copy(v)
        w[0]=1
        return w
    else:
        return v/(np.sum(v))

def sample_parameters(n_clusters, dim, **params_gen):

    parameters_gen={}
    parameters_gen['cluster_prop'] = normalize(np.random.poisson(10,n_clusters))
    for cluster in np.arange(n_clusters):
        parameters_gen['mean'+str(cluster)] = np.random.multivariate_normal(mean=np.zeros(dim), cov = params_gen.get('spread_means',10)*np.identity(dim)) 
        parameters_gen['cov'+str(cluster)] = datasets.make_spd_matrix(n_dim = dim) + params_gen.get('extra_cor', 0)*np.ones(shape=(dim,dim))
    parameters_gen.update(params_gen)
    
    return parameters_gen    



class GMM_X(object): 
    
    def __init__(self, num_samples, num_dim, num_clusters, **params_gen):
        
        self.n_samples = num_samples   
        self.dim = num_dim
        self.n_clusters = num_clusters               

        self.parameters = sample_parameters(n_clusters = self.n_clusters, dim = self.dim, **params_gen)
                                                            
        self.sample()        
        
    def sample(self):

        self.X = np.empty((self.n_samples, self.dim))
        self.assignments = np.random.choice(self.n_clusters,size=self.n_samples,p=self.parameters['cluster_prop'])
        for cluster in np.arange(self.n_clusters):
            self.X[np.where(self.assignments == cluster)[0],:] = np.random.multivariate_normal(mean = self.parameters['mean'+str(cluster)], cov = self.parameters['cov'+str(cluster)], size = np.sum(self.assignments == cluster) ) 
    
    