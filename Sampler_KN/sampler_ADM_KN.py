import numpy as np
import scipy as sp
import pystan as ps
        
class ADM_KN(object):
    
    def __init__(self, num_populations_KN, multiKN = 1, fastStructureStan = None, fastStructureSampleStan = None, **sampler_kn_args):
        
        self.num_populations_KN = num_populations_KN
        self.multiKN = multiKN  
        self.alpha =  sampler_kn_args.get('alpha', np.ones(self.num_populations_KN))
        self.a_beta = sampler_kn_args.get('a_beta', 1)
        self.b_beta = sampler_kn_args.get('b_beta', 1)
        self.fastStructureStan = fastStructureStan if fastStructureStan is not None else ps.StanModel('/home/roquero/Code_KN/Sampler_KN/Utils/fastStructure.stan')
        self.fastStructureSampleStan = fastStructureSampleStan if fastStructureSampleStan is not None else ps.StanModel('/home/roquero/Code_KN/Sampler_KN/Utils/fastStructureSample.stan')

    def fit(self, X):
        self.X = X                                                                         
        self.n_samples, self.dim = self.X.shape[0], self.X.shape[1]   
        self.estimate_stan = {'N':self.n_samples, 'K':self.n_populations_KN, 'L':self.dim, 'alpha':self.alpha, 'b1':self.a_beta, 'b2':self.b_beta, 'G': self.X}
        self._estimate()
        
        
    def sample(self, X_new = None):   
        if X_new is not None:
            assert X_new.shape[1] == self.dim, 'Different input dimensions when fitting and sampling.'
            self._sample(X_new)
        else:
            self._sample(self.X)
        return self.X_KN
        
    def _estimate(self, **params_hyperprior):
        self.optimal_params = self.fastStructureStan.optimizing(data = self.estimate_stan)
        self.P_hat = self.optimal_params['P']
        
    def _sample(self, X_sample):
        self.sample_stan = {'N':self.n_samples, 'K':self.num_populations_KN, 'L':self.dim, 'alpha':self.alpha, 'P': self.P_hat, 'G': X_sample}
        self.stan_samples = self.fastStructureSampleStan.sampling(data = self.sample_stan, iter = 2000)
        self.Q_hat = self.stan_samples.extract('Q')['Q'][0]
        self.knockoffADM = ADM_Snip(self.dim, self.num_populations_KN, X_sample.shape[0], P_gen = self.P_hat, Q_gen = self.Q_hat)
        self.X_KN = self.knockoffADM.X
