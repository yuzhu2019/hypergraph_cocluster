"""
Author: Yu Zhu, Boning Li, Rice ECE
References:
    [1] Hypergraph Random Walks, Laplacians, and Clustering
"""
import numpy as np
import scipy
from tqdm.auto import trange, tqdm
import warnings

def comp_hyperedge_weights(R):
    """
    :param R: weighted incidence matrix of shape |E| x |V| containing non-negative entries
    :return: hyperedge weights computed as std of every row of R
    """
    hyperedge_weights = np.std(R, axis=1)
    return hyperedge_weights


def comp_W(R, hyperedge_weights):
    """
    :param R: weighted incidence matrix of shape |E| x |V| containing non-negative entries
    :param hyperedge_weights: weight of every hyperedge
    :return: hyperedge-weight matrix of shape |V| x |E|
    """
    num_edges, num_nodes = np.shape(R)
    hw_mat = np.tile(hyperedge_weights, (num_nodes,1))
    W = np.multiply(R.T>0, hw_mat)
    return W


def comp_P(R, W, alpha=1):
    """
    :param R: weighted incidence matrix of shape |E| x |V| containing non-negative entries
    :param W: hyperedge-weight matrix of shape |V| x |E|
    :param alpha: hyperparameter for laziness interpolation in random walks
    :return: transition probability matrix defined in Equation (3) in [1]
    """
    assert R.shape == W.T.shape
    
    if alpha==1:
        Dv_inv = np.diag(1 / W.sum(axis=1))
        De_inv = np.diag(1 / R.sum(axis=1))
        P = Dv_inv @ W @ De_inv @ R  
    
    else:
        n_e, n_v = R.shape
        dv = W.sum(axis=1)
        de = R.sum(axis=1)
        P = np.zeros((n_v, n_v))
        for u in trange(n_v):
            Wu = W[u,:]
            Ru = R[:,u]
            dvu = dv[u]
            for v in range(n_v):
                Rv = R[:,v]
                E_uv = np.logical_and(Ru,Rv) # u in e and v in e   
                p = np.sum(Wu[E_uv]*Rv[E_uv]/(de[E_uv]-(1-alpha)*Ru[E_uv]))
                P[u,v] = p/dvu * alpha**(v==u)
    return P


def comp_pi(P):
    """
    :param P: transition probability matrix defined in Equation (3) in [1]
    :return: stationary distribution of random walks defined by P
    """
    _, vl = scipy.linalg.eig(P, left=True, right=False)
    pi = np.real(vl[:, 0])  # the all-positive dominant left eigenvector of P
    pi = pi / np.sum(pi)  # unit 1-norm
#    assert np.sum(abs(pi @ P - pi)) < 1e-10  # check Equation (4) in [1]
    if not np.allclose(pi @ P, pi):
        warnings.warn("check Equation (4) in [1], not all close: %.6e"%np.sum(abs(pi @ P - pi)))
    return pi


def comp_L(P, pi):
    """
    :param P: transition probability matrix defined in Equation (3) in [1]
    :param pi: stationary distribution of random walks defined by P
    :return: the directed combinatorial Laplacian defined in Equation (5) in [1]
    """
    return np.diag(pi) - (np.reshape(pi, (-1, 1)) * P + P.T * pi) / 2


def normalize_L(L, pi):
    """
    :param L: the directed combinatorial Laplacian defined in Equation (5) in [1]
    :param pi: stationary distribution of random walks defined by P
    :return: the normalized Laplacian defined in Equation (6) in [1]
    """
    pi_sr = pi ** -0.5
    return np.reshape(pi_sr, (-1, 1)) * L * pi_sr


def comp_T(normalized_L):
    """
    :param normalized_L: the normalized Laplacian defined in Equation (6) in [1]
    :return: the matrix in Equation (15) in [1]
    """
    return np.eye(len(normalized_L)) - normalized_L


