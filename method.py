"""
References:
    [1] Hypergraph Random Walks, Laplacians, and Clustering
"""

import numpy as np
from scipy.linalg import eigh
from scipy.linalg import svd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils import *
from algs import clique_expansion as cx
from algs import star_expansion as sx


def clique_laplacian(X, k, alpha=1, verbose=False):
    print('comp_R...', verbose=verbose)
    try:
        R = X.T.toarray()
    except:
        R = X.T

    print('comp_hyperedge_weights...', verbose=verbose)
    hyperedge_weights = cx.comp_hyperedge_weights(R)

    print('comp_W...', verbose=verbose)
    W = cx.comp_W(R, hyperedge_weights)

    print('comp_P...', verbose=verbose)
    P = cx.comp_P(R, W, alpha)

    print('comp_pi...', verbose=verbose)
    pi = cx.comp_pi(P)

    print('comp_L...', verbose=verbose)
    L = cx.comp_L(P, pi)

    print('normalize_L...', verbose=verbose)
    normalized_L = cx.normalize_L(L, pi)

    print('comp_T...', verbose=verbose)
    T = cx.comp_T(normalized_L)
    
    # Compute the 𝑘 eigenvectors paired with the 𝑘 largest eigenvalues of T 
    #and collect them into the columns of the matrix U ∈ R|𝑉|×𝑘.
    N = T.shape[0]
    _, U = eigh(T, subset_by_index=[N-k, N-1])
    U = normalize(np.real(U), norm='l2', axis=1)
    
    return U


def star_laplacian(X, k, alg='alg1', verbose=False):
    print('comp_R...', verbose=verbose)
    try:
        R = X.T.toarray()
    except:
        R = X.T
        
    print('comp_hyperedge_weights...', verbose=verbose)
    hyperedge_weights = sx.comp_hyperedge_weights(R)

    print('comp_W...', verbose=verbose)
    W = sx.comp_W(R, hyperedge_weights)

    print('comp_P_VE...', verbose=verbose)
    P_VE = sx.comp_P_VE(W)

    print('comp_P_EV...', verbose=verbose)
    P_EV = sx.comp_P_EV(R)

    print('comp_P...', verbose=verbose)
    P = sx.comp_P(P_VE, P_EV)

    print('comp_P_alpha...', verbose=verbose)
    P_alpha = sx.comp_P_alpha(P)

    print('comp_pi...', verbose=verbose)
    pi = sx.comp_pi(P_alpha)

    print('comp_A_bar...', verbose=verbose)
    A_bar = sx.comp_A_bar(P_VE, P_EV, pi)

    print('comp_V...', verbose=verbose)
    V = sx.comp_V(A_bar, k)
    
    print(alg, '...', verbose=verbose)
    if alg=='alg1':
        V1 = sx.alg1(V, pi)  # contain both vertices and hyperedges
        return V1
    elif alg=='alg2':
        V2 = sx.alg2(V)
        return V2
    else:
        raise

def spectal_bicluster(X, k, verbose=False):
    # bipartite spectral graph partitioning
    try:
        R = X.T.toarray() # word-by-document matrix
    except:
        R = X.T  
    D1 = np.diag(np.sum(R, 1) ** -0.5)  
    D2 = np.diag(np.sum(R, 0) ** -0.5)  
    Rn = D1 @ R @ D2
    U, s, Vh = svd(Rn)
    V_w = D1 @ U[:, :k]  # for words
    V_d = D2 @ Vh[:k, :].T  # for documents
    normalized_features = np.block([[V_d],[V_w]])
    return normalized_features
        
      
        
class Cluster(object):
    def __init__(self, method, k, **kwargs):
        """
        An implementation of Algorithm 1 (steps 4-6) in [1].

        :param k: number of clusters
        :param method: which method - {star_expansion, clique_expansion, pca, sbc, raw}
        :param kwargs: named arguments -
            verbose: control print. set True if you want to print info during the computations
            seed: random seed for kmeans clustering
            ncomponents: [method=pca] how many pc's to compute in the first place. default=k 
            alpha: [method=clique_expansion] random walk interpolation factor. default=1
            alg: [method=star_expansion] which algorithm to use for laplacian  
            proba (obs): whether to use probalistic clustering. True --> GMM; False --> kmeans
            separate: if true, cluster docs and words separately; else, cluster them together. default=False

        :return labels_pred: predicted class labels
        :return normalized_U: normalized matrix of shape |V| x k columns being k largest eigenvectors

        Adapted from:
        [1] https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
        """        
        
        self.method = method
        self.k = k
        self.vb = 'verbose' in kwargs and kwargs['verbose']
        self.seed = kwargs['seed'] if 'seed' in kwargs else 42
#         self.proba = kwargs['proba'] if 'proba' in kwargs else False
        self.sepa = kwargs['separate'] if 'separate' in kwargs else False  
        
        # initialize clusters
        self.cluster = [KMeans(n_clusters=k, random_state=self.seed)]
        if self.sepa:
            self.cluster.append(KMeans(n_clusters=k, random_state=self.seed))
        self.features = []
        self.ndoc, self.nword = None, None
        
    def compute_features(self, X, k, alpha):
        normalized_features, norm_feat_doc, norm_feat_word = None, None, None
        
        if self.method.startswith('cliq'):
            assert self.sepa
            norm_feat_doc = clique_laplacian(X, k, alpha, verbose=self.vb)
            norm_feat_word = clique_laplacian(X.T, k, alpha, verbose=self.vb)

        elif self.method.startswith('star'):
            alg = self.method.split('_')[-1]
            normalized_features = star_laplacian(X, k, alg=alg, verbose=self.vb)

        elif self.method == 'sbc':
            normalized_features = spectal_bicluster(X, k, verbose=self.vb)        

        elif self.method == 'pca':
            pca = PCA(n_components=k) # Create a PCA instance: pca

            # Standardize the data to have a mean of ~0 and a variance of 1
            X_std = StandardScaler().fit_transform(X.toarray()) 
            norm_feat_doc = pca.fit_transform(X_std)[:,:k]

            X_std_t = StandardScaler().fit_transform(X.T.toarray())
            norm_feat_word = pca.fit_transform(X_std_t)[:,:k]  

        elif self.method == 'raw':
            assert self.sepa
            # Standardize the data to have a mean of ~0 and a variance of 1
            norm_feat_doc = StandardScaler().fit_transform(X.toarray())
            norm_feat_word = StandardScaler().fit_transform(X.T.toarray())

        else:
            raise NotImplemented
            
        return normalized_features, norm_feat_doc, norm_feat_word
            
    
    def get_raw_labels(self):
        return np.concatenate([cls.labels_ for cls in self.cluster])
    
    
    def fit_predict(self, X, **kwargs):
        
        alpha=kwargs['alpha'] if 'alpha' in kwargs else 1
        
        ndoc, nword = X.shape
        self.ndoc, self.nword = ndoc, nword
        
        normalized_features, norm_feat_doc, norm_feat_word = self.compute_features(X, self.k, alpha)

        if not self.sepa:
            if normalized_features is None:
                normalized_features = np.vstack((norm_feat_doc, norm_feat_word))
            self.features = [normalized_features]
        else:
            if norm_feat_doc is None:
                norm_feat_doc = normalized_features[:ndoc]
                norm_feat_word = normalized_features[ndoc:]
            self.features = [norm_feat_doc, norm_feat_word]
            
        for i, feat in enumerate(self.features):
            self.cluster[i] = self.cluster[i].fit(feat)

        labels = self.get_raw_labels()
        return labels
    
    
    def wcenters_matched(self, matched_wlabels):
        # raw label mapping to assigned label
        wlabel_map = {k[0]:k[1] for k in np.vstack((matched_wlabels, self.get_raw_labels()[-self.nword:])).T}
        # reorder the centers to match assigned labels
        centers = self.cluster[-1].cluster_centers_[[p[1] for p in sorted(wlabel_map.items())]]
        return centers
    
    def dcenters_matched(self, matched_dlabels):
        # raw label mapping to assigned label
        dlabel_map = {k[0]:k[1] for k in np.vstack((matched_dlabels, self.get_raw_labels()[:self.ndoc])).T}
        # reorder the centers to match assigned labels
        centers = self.cluster[0].cluster_centers_[[p[1] for p in sorted(dlabel_map.items())]]
        return centers
    
    
    def word_similarity(self, nword, method='cosine'):
        word_features = self.features[-1][-nword:]
        word_labels = self.cluster[-1].labels_[-nword:]
        word_centers = self.cluster[-1].cluster_centers_ 
        
        word_similarity = np.zeros(nword)
        
        if method == 'cosine':
            func = cosine_similarity
        elif method == 'gaussian':
            func = euclidean_distances
        else:
            raise NotImplemented
        
        for wl, wc in zip(np.unique(word_labels), word_centers):
            bi = word_labels==wl
            word_similarity[bi] = func(word_features[bi], wc.reshape(1, -1)).flatten()
            
        if method == 'gaussian':
            word_similarity = np.exp(-word_similarity / word_similarity.std()) # or choose an s
    
        return word_similarity
        
        
        
def clustering(X, k, method, **kwargs):
    """
    An implementation of Algorithm 1 (steps 4-6) in [1].
    
    :param X: feature matrix of shape |V| x |E|
    :param k: number of clusters
    :param method: which method - {star_expansion, clique_expansion, pca, sbc, raw}
    :param kwargs: named arguments -
        verbose: control print. set True if you want to print info during the computations
        seed: random seed for kmeans clustering
        ncomponents: [method=pca] how many pc's to compute in the first place. default=k 
        alpha: [method=clique_expansion] random walk interpolation factor. default=1
        alg: [method=star_expansion] which algorithm to use for laplacian  
        proba: whether to use probalistic clustering. True --> GMM; False --> kmeans
        separate: if true, cluster docs and words separately; else, cluster them together. default=False
    
    :return labels_pred: predicted class labels
    :return normalized_U: normalized matrix of shape |V| x k columns being k largest eigenvectors
    
    Adapted from:
    [1] https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
    """
    
    vb = 'verbose' in kwargs and kwargs['verbose']
    seed = kwargs['seed'] if 'seed' in kwargs else 42
    proba = kwargs['proba'] if 'proba' in kwargs else False
    sepa = kwargs['separate'] if 'separate' in kwargs else False
    
    ndoc, nword = X.shape
    normalized_features, norm_feat_doc, norm_feat_word = None, None, None
    
    if method.startswith('cliq'):
        assert sepa
        alpha=kwargs['alpha'] if 'alpha' in kwargs else 1
        norm_feat_doc = clique_laplacian(X, k, alpha, verbose=vb)
        norm_feat_word = clique_laplacian(X.T, k, alpha, verbose=vb)
            
    elif method.startswith('star'):
        normalized_features = star_laplacian(X, k, alg=kwargs['alg'], verbose=vb)
                
    elif method == 'sbc':
        normalized_features = spectal_bicluster(X, k, verbose=vb)        
        
    elif method == 'pca':
        nc = kwargs['ncomponents'] if 'ncomponents' in kwargs else k
        assert nc >= k
        
        pca = PCA(n_components=nc) # Create a PCA instance: pca
        
        # Standardize the data to have a mean of ~0 and a variance of 1
        X_std = StandardScaler().fit_transform(X.toarray()) 
        norm_feat_doc = pca.fit_transform(X_std)[:,:k]
        
        X_std_t = StandardScaler().fit_transform(X.T.toarray())
        norm_feat_word = pca.fit_transform(X_std_t)[:,:k]  
                            
    elif method == 'raw':
        assert sepa
        # Standardize the data to have a mean of ~0 and a variance of 1
        norm_feat_doc = StandardScaler().fit_transform(X.toarray())
        norm_feat_word = StandardScaler().fit_transform(X.T.toarray())
                        
    else:
        raise NotImplemented
        
    # clustering
    if not sepa:
        if normalized_features is None:
            normalized_features = np.vstack((norm_feat_doc, norm_feat_word))
            
        if not proba: 
            # Cluster the rows of the matrix U using 𝑘-means.
            kmeans = KMeans(n_clusters=k, random_state=seed).fit(normalized_features)
            labels_pred = kmeans.labels_ 
            return labels_pred, normalized_features
        else: 
            # probalistic clustering by GMM-EM
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=seed).fit(normalized_features)
            labels_pred = gmm.predict(normalized_features)
            labels_prob = gmm.predict_proba(normalized_features)
            return (labels_pred, labels_prob), (normalized_features, gmm)
        
    else:
        if norm_feat_doc is None:
            norm_feat_doc = normalized_features[:ndoc]
            norm_feat_word = normalized_features[ndoc:]
        
        if not proba:
            kmeans_doc = KMeans(n_clusters=k, random_state=seed).fit(norm_feat_doc)
            kmeans_word = KMeans(n_clusters=k, random_state=seed).fit(norm_feat_word)
            labels_pred = np.concatenate((kmeans_doc.labels_, kmeans_word.labels_))
            return labels_pred, (norm_feat_doc, norm_feat_word)
        else:
            gmm_doc = GaussianMixture(n_components=k, covariance_type='full', random_state=seed).fit(norm_feat_doc)
            gmm_word = GaussianMixture(n_components=k, covariance_type='full', random_state=seed).fit(norm_feat_word)
            labels_pred = np.concatenate((gmm_doc.predict(norm_feat_doc), gmm_word.predict(norm_feat_word)))
            labels_prob = np.concatenate((gmm_doc.predict_proba(norm_feat_doc), gmm_word.predict_proba(norm_feat_word)))
            return (labels_pred, labels_prob), ((norm_feat_doc, norm_feat_word), (gmm_doc, gmm_word))
    



