"""
Author: Yu Zhu, Boning Li, Rice ECE
"""
import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import svd

from .clique_expansion import comp_hyperedge_weights,comp_W, comp_pi


def comp_P_VE(W):
    return normalize(W, norm='l1', axis=1)

def comp_P_EV(R):
    return normalize(R, norm='l1', axis=1)

def comp_P(P_VE, P_EV):
    num_edges, num_nodes = np.shape(P_EV)
    P = np.block([[np.zeros((num_nodes, num_nodes)), P_VE],[P_EV, np.zeros((num_edges, num_edges))]])
    return P

def comp_P_alpha(P, alpha=0.5):
    return (1 - alpha) * np.eye(len(P)) + alpha * P

def comp_A_bar(P_VE, P_EV, pi):
    num_edges, num_nodes = np.shape(P_EV)
    pi_1 = pi[:num_nodes]
    pi_2 = pi[-num_edges:]
    return (np.diag(pi_1**0.5) @ P_VE @ np.diag(pi_2**-0.5) + np.diag(pi_1**-0.5) @ P_EV.T @ np.diag(pi_2**0.5))/2

def comp_V(A_bar, k):
    num_nodes, num_edges = np.shape(A_bar)
    assert k <= min(num_nodes, num_edges)
    U_bar, s_bar, Vh_bar = svd(A_bar)
    V_V = U_bar[:, :k]  
    V_E = Vh_bar[:k, :].T
    V = np.block([[V_V],[V_E]])
    return np.real(V)
    
def alg1(V, pi):
    return np.diag(pi**-0.5) @ V

def alg2(V):
    return normalize(V, norm='l2', axis=1)

