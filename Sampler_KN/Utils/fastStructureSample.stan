
data {                      // Data block
  int<lower=1> N;           // Sample size
  int<lower=1> K;           // Population size
  int<lower=1> L;           // Loci size
  int<lower=0,upper=2> G[N,L];           // SNP Matrix
  matrix<lower=0,upper=1>[K,L] P;           // Coefficient vector
  vector<lower=0>[K] alpha;
}

transformed data {          // Transformed data block. Not used presently.
  matrix[K,K] log_categ[L,3];
  
  for (ka in 1:K){
    for (kb in 1:K){
      for (l in 1:L){
        log_categ[l,1][ka,kb] = log(1-P[ka,l])+ log(1-P[kb,l]);
        log_categ[l,2][ka,kb] = log(P[ka,l]*(1-P[kb,l])+ (1-P[ka,l])*P[kb,l]);
        log_categ[l,3][ka,kb] = log(P[ka,l]) + log(P[kb,l]);
      }
    }
  }

} 

parameters {                // Parameters block  
  simplex[K] Q[N];          // Admixture parameters
}

transformed parameters {
  matrix[K,K] log_Q[N];
  
  for (n in 1:N){
     log_Q[n] = rep_matrix(log(Q[n]),K) + rep_matrix(log(Q[n]),K)';
  }
  
}

model {                     // Model block

  // priors
  for (n in 1:N){
    Q[n] ~ dirichlet(alpha);
  }

  // likelihood
  for (n in 1:N){    
    for (l in 1:L){
      target += log_sum_exp(log_Q[n] + log_categ[l,(G[n,l]+1)]);
    }
  }
}
generated quantities {      // Generated quantities block. Not used presently.
}
