from SNPknock import knockoffHMM, models       
import SNPknock.fastphase as fp

import numpy as np
import scipy as sp

class HMM_KN(object):
    
    def __init__(self, num_hidden_states_kn, numit = 25, multiKN = 1, **sampler_kn_args):
        
        self.hidden_states = num_hidden_states_kn
        self.multiKN = multiKN  
        self.numit = numit
                
    def fit(self, X):
        self.X = X                                                                         
        self.n_samples, self.dim = self.X.shape[0], self.X.shape[1]            
        self._estimate()
        
        
    def sample(self, X_new = None):   
        if X_new is not None:
            assert X_new.shape[1] == self.dim, 'Different input dimensions when fitting and sampling.'
            self._sample(X_new)
        else:
            self._sample(self.X)
        return self.X_KN
        
    
    def _estimate(self, Xfp_file='./X.inp'):
        fp.writeX(self.X, Xfp_file)
        path_to_fp = "/home/roquero/Software/fastPHASE" # Relative path to the fastPhase executable
        out_path = "./example" # Prefix to temporary output files produced by fastPhase
        fp.runFastPhase(path_to_fp, Xfp_file, out_path, K = self.hidden_states, numit = self.numit)
        r_file     = out_path + "_rhat.txt"
        alpha_file = out_path + "_alphahat.txt"
        theta_file = out_path + "_thetahat.txt"  
        self.hmm = fp.loadFit(r_file, theta_file, alpha_file, self.X[0,:])
        self.knockoffHMM = knockoffHMM(self.hmm["pInit"], self.hmm["Q"], self.hmm["pEmit"])    
            
    def _sample(self, X_sample):
        self.X_KN = self.knockoffHMM.sample(X_sample)