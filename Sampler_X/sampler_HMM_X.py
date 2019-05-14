from SNPknock import models 
import numpy as np

def sample_parameters(dim, n_hidden_states, n_emission_states):
    p = dim
    K = n_hidden_states
    M = n_emission_states
    Q = np.zeros((p-1,K,K))
    for j in range(p-1):
       Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
       Q[j,:,:] += np.diag([10]*K)
       Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
    pEmit = np.zeros((p,M,K))
    for j in range(p):
      pEmit[j,:,:] = np.resize(np.random.uniform(size=M*K),(M,K))
      pEmit[j,:,:] /= np.sum(pEmit[j,:,:],0)
    pInit = np.array([1.0/K]*K)
    parameters_gen = {'Q':Q, 'pEmit': pEmit, 'pInit': pInit}
    
    return parameters_gen

class HMM_X(object):
    
    def __init__(self, num_samples, num_dim, num_hidden_states, num_emission_states = 3, **sampler_x_args):
        self.n_samples = num_samples
        self.dim = num_dim
        self.n_hidden_states = num_hidden_states
        self.n_emission_states = num_emission_states
        
        self.parameters = sample_parameters(self.dim, self.n_hidden_states, self.n_emission_states)
        
        self.modelX = models.HMM(self.parameters['pInit'], self.parameters['Q'], self.parameters['pEmit'])
        
        self.sample()
        
    def sample(self):
        self.X = self.modelX.sample(self.n_samples)

