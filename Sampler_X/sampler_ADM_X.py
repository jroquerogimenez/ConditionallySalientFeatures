import numpy as np

def sample_parameters(n_samples, dim, n_populations, **params_gen):

    alpha = params_gen.get('alpha', np.ones(n_populations))
    a_beta, b_beta = params_gen.get('a_beta', 1), params_gen.get('b_beta', 1)

    Q_gen = np.random.dirichlet(alpha = alpha, size = n_samples)
    P_gen = np.random.beta(a = a_beta, b = b_beta, size = (n_populations, dim))
    parameters_gen = {'Q_gen': Q_gen, 'P_gen': P_gen}
    parameters_gen.update(params_gen)
    
    return parameters_gen
    
        
class ADM_X(object):
    
    def __init__(self, num_samples, num_dim, n_populations, **params_gen):
        self.n_samples = num_samples
        self.dim = num_dim
        self.n_populations = n_populations
        
        self.parameters = sample_parameters(self.n_samples, self.dim, self.n_populations, **params_gen)

        self.sample()
        
    
    def sample(self):
        self.P_gen = self.parameters['P_gen']
        self.Q_gen = self.parameters['Q_gen']
        self.Za_gen = np.sum(np.random.uniform(
            size = (self.dim, self.n_samples, 1)) > np.cumsum(self.Q_gen,axis = 1), axis = 2).T
        self.Zb_gen = np.sum(np.random.uniform(
            size = (self.dim, self.n_samples, 1)) > np.cumsum(self.Q_gen,axis = 1), axis = 2).T
        self.X = np.random.binomial(1,self.P_gen[self.Za_gen,np.arange(self.dim)]) + np.random.binomial(1,self.P_gen[self.Zb_gen,np.arange(self.dim)])

