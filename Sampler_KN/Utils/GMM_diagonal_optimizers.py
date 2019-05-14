import numpy as np
import scipy as sp
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
solvers.options['maxiters'] = 100
solvers.options['feastol'] = 1e-7

def flatten_Ei(index,dim):
    Ei = np.zeros((dim,dim))
    Ei[index,index]=1
    return Ei.flatten()


def create_diagonal(cov_matrix, multiKN_par=2, method_diagonal='MI_diagonal_plus'):
    if method_diagonal == 'SDP_diagonal':
        diagonal = SDP_diagonal(cov_matrix, multiKN_par)
    elif method_diagonal == 'MI_diagonal':
        diagonal = MI_diagonal(cov_matrix, multiKN_par)
    elif method_diagonal == 'EQUI_diagonal':
        diagonal = EQUI_diagonal(cov_matrix, multiKN_par)
    elif method_diagonal == 'MI_diagonal_plus':
        diagonal = MI_diagonal_plus(cov_matrix, multiKN_par)
    else:
        print('Method not implemented')
    return diagonal


def SDP_diagonal(cov_matrix, multiKN_par):
    cor_matrix = np.diag(1/np.sqrt(np.diag(cov_matrix))).dot(cov_matrix.dot(np.diag(1/np.sqrt(np.diag(cov_matrix)))))
    dim = cov_matrix.shape[0]
    c = matrix(-np.ones(dim))
    Gl = matrix(np.vstack([-np.identity(dim), np.identity(dim)]))
    hl = matrix(np.hstack([np.zeros(dim), np.ones(dim)]))
    Gs = [matrix(np.vstack([flatten_Ei(l,dim) for l in np.arange(dim)]).T)]
    hs = [matrix(multiKN_par*cor_matrix)]            
    result = solvers.sdp(c,Gl=Gl,hl=hl,Gs=Gs,hs=hs)
    return np.diag((np.array(result['x'])[:,0]))*0.9999*np.diag(cov_matrix)


def EQUI_diagonal(cov_matrix, multiKN_par):
    cor_matrix = np.diag(1/np.sqrt(np.diag(cov_matrix))).dot(cov_matrix.dot(np.diag(1/np.sqrt(np.diag(cov_matrix)))))
    eigenvalues, _ = np.linalg.eig(cor_matrix)
    value = np.amin([multiKN_par*np.amin(eigenvalues),1])
    return np.diag(value*np.diag(cov_matrix))

def MI_diagonal(cov_matrix, multiKN_par):
    dim = cov_matrix.shape[0]
    min_eigval = np.maximum(np.real(np.amin(np.linalg.eig(cov_matrix)[0]))/dim,1e-20)
    multiKN = 1/(multiKN_par - 1)
    def opt_function(x=None, z=None):
        if x is None: 
            return 0, matrix(np.reshape(min_eigval*np.ones(dim),(dim,1)))
        if np.amin(x) <= 0: 
            return None
        u = np.array(x).reshape(dim)
        aux_matrix = np.linalg.inv(multiKN_par*cov_matrix-np.diag(u))
        if z is None:
            return objective(u), matrix(grad_objective(u, aux_matrix))
        else:
            return objective(u), matrix(grad_objective(u, aux_matrix)), z*matrix(hess_objective(u, aux_matrix))   
    def objective(u):           
        return -np.log(np.linalg.det(multiKN_par*cov_matrix - np.diag(u))) - multiKN*np.sum(np.log(u)) 
    def grad_objective(u, aux_matrix):
        return np.reshape( np.diag(aux_matrix) - (1/u)*multiKN, (1,dim) )
    def hess_objective(u, aux_matrix):          
        return np.square(aux_matrix) + np.diag(np.square(1/u))*multiKN     
    
    dims={'l': 0, 'q': [], 's': [dim]}
    G = matrix(np.vstack([flatten_Ei(l,dim) for l in np.arange(dim)]).tolist())
    h = matrix(list(multiKN_par*cov_matrix.flatten()))       
    solved = False
    while not solved:
        try:
            result = solvers.cp(F=opt_function, G=G, h=h,dims = dims)
            solved = True
        except ValueError:
            print('Relaxing feasible set')
            solvers.options['feastol'] *= 10
    return np.diag((np.array(result['x'])[:,0]))

def MI_diagonal_plus(cov_matrix, multiKN_par):
    MI_diag = MI_diagonal(cov_matrix, multiKN_par)
    increase = True
    coef = 1.01
    while increase:
        try:
            np.linalg.cholesky(multiKN_par*cov_matrix- coef*MI_diag)
            coef *= 1.01
            print('Increasing diagonal')
        except np.linalg.LinAlgError:
            increase = False
            coef /= 1.01
    if coef == 1: print('No increase w.r.t. MI diagonal.')
    return coef*MI_diag




