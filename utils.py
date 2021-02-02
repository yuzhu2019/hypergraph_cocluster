import sys
import os
from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import jaccard_score, f1_score, normalized_mutual_info_score, confusion_matrix, adjusted_rand_score, accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as patches
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy import sparse 
from wordcloud import WordCloud 
from adjustText import adjust_text


try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

def print(*args, **kwargs):
    """My custom print() function.
    Adapted from https://stackoverflow.com/questions/550470/overload-print-python
    """
    # Adding new arguments to the print function signature 
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    
    if 'verbose' in kwargs and not kwargs['verbose']:
        return None
    else:
        #__builtin__.print('My overridden print() function!')
        kwargs.pop('verbose', None)
        return __builtin__.print(*args, **kwargs)


def display_words(words, labels, importance, categories, ax, ntop=5, pos=None):
    """
    Draw wordcloud based on word frequencies.
    :param words: list of vocabs
    :param labels: list of labels corresponding to words
    :param importance: list of importance (e.g. count/freq) corresponding to words
    :param categories: list of categories
    :param ax: axes to plot wordcloud images
    """
    colors = ['orange','red','royalblue','green', 'magenta']#plt.rcParams['axes.prop_cycle'].by_key()['color']
    words = np.array(words)
    importance = np.array(importance)
    
    xy_init = [[-45,-30],
               [-40,50],
               [20,-40],
               [50,-10],
               [20,50]]
       
    for ci, ci_str in zip(np.unique(labels), categories):

        ii = labels==ci
        idx = importance[ii].argsort()[-ntop:][::-1]
        word_disp = words[ii][idx]
        word_posi = pos[ii][idx]
        
        print(word_disp)

        tx,ty = xy_init[ci]
        for wi, (wd,wp) in enumerate(zip(word_disp, word_posi)):
            px, py = wp
            ha = 'right' if tx<px else 'left'
            va = 'top' if ty<py else 'bottom'
            plt.text(tx, ty-wi*3, wd, ha=ha, va=va)
            ax.plot([tx, px],[ty-wi*3, py], color=colors[ci], linewidth=.3)
            
            
        
def show_word_cloud(words, labels, importance, categories, axes, seed=42):
    """
    Draw wordcloud based on word frequencies.
    :param words: list of vocabs
    :param labels: list of labels corresponding to words
    :param importance: list of importance (e.g. count/freq) corresponding to words
    :param categories: list of categories
    :param axes: axes to plot wordcloud images
    """
    words = np.array(words)
    importance = np.array(importance)
    
    for ci, ci_str in zip(np.unique(labels), categories):

        ii = labels==ci
        size_dict = dict(zip(words[ii], importance[ii]))

        # Generate a word cloud image
        wordcloud = WordCloud(background_color="white", random_state=seed)
        wordcloud.fit_words(size_dict)

        # Display the generated image:
        axes[ci].imshow(wordcloud, interpolation='bilinear')
        axes[ci].set_axis_off()
        axes[ci].set_title(ci_str)   
    
    
def word_freq_inclass(count_mat, labels_pred):
    ndoc, nword = count_mat.shape
    dlabels = labels_pred[:ndoc]
    wlabels = labels_pred[ndoc:]
    
    wfreq = np.zeros_like(wlabels).astype(float)
    for w, wl in enumerate(wlabels):
        idx = dlabels==wl
        wfreq[w] = count_mat[idx,w].sum()
                    
    for wl in np.unique(wlabels):
        idx = wlabels==wl
        wfreq[idx] *= 1./wfreq[idx].sum()
    
    return wfreq

    
def munkres_assignment(pred, true, verbose=False):
    """
    Assign cluster labels by the Hungarian method. Adapted from https://stackoverflow.com/questions/55258457/find-mapping-that-translates-one-list-of-clusters-to-another-in-python
    :param pred: predicted cluster indices
    :param true: true cluster indices
    :return matched: reordered label predictions
    """
    m = Munkres()
    contmat = contingency_matrix(pred, true)
    mdict = dict(m.compute(contmat.max() - contmat))
    matched = np.sum([(pred == p)*t for p, t in mdict.items()], axis=0)
    if verbose:
        print("matching cluster labels:", ", ".join(["pred %d --> %d"%(p, t) for p,t in mdict.items()]),'\n')
    return matched


def set_row_csr(A, row_idx, new_row):
    '''
    Replace a row in a CSR sparse matrix A.

    Parameters
    ----------
    A: csr_matrix
        Matrix to change
    row_idx: int
        index of the row to be changed
    new_row: np.array
        list of new values for the row of A

    Returns
    -------
    None (the matrix A is changed in place)

    Prerequisites
    -------------
    The row index shall be smaller than the number of rows in A
    The number of elements in new row must be equal to the number of columns in matrix A
    '''
    assert sparse.isspmatrix_csr(A), 'A shall be a csr_matrix'
    assert row_idx < A.shape[0], \
            'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
            .format(row_idx, A.shape[0])
    try:
        N_elements_new_row = len(new_row)
    except TypeError:
        msg = 'Argument new_row shall be a list or numpy array, is now a {0}'\
        .format(type(new_row))
        raise AssertionError(msg)
    N_cols = A.shape[1]
    assert N_cols == N_elements_new_row, \
            'The number of elements in new row ({0}) must be equal to ' \
            'the number of columns in matrix A ({1})' \
            .format(N_elements_new_row, N_cols)

    idx_start_row = A.indptr[row_idx]
    idx_end_row = A.indptr[row_idx + 1]
    additional_nnz = N_cols - (idx_end_row - idx_start_row)

    A.data = np.r_[A.data[:idx_start_row], new_row, A.data[idx_end_row:]]
    A.indices = np.r_[A.indices[:idx_start_row], np.arange(N_cols), A.indices[idx_end_row:]]
    A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):] + additional_nnz]
    

def eval_summary(matched, true, categories=None):
    """
    Evaluation metrics: nmi, f1, jac, etc. 
    :param matched: matched predicted cluster indices
    :param true: true cluster indices
    :param categories: cluster names
    """

    acc = accuracy_score(true, matched)
    
    # normalized mutual information (NMI)
    nmi = normalized_mutual_info_score(true, matched)

    # F1
#     f1_macro = f1_score(true, matched, average='macro')
    f1_weighted = f1_score(true, matched, average='weighted')

    # Jaccard
#     jac_macro = jaccard_score(true, matched, average='macro')
#     jac_weighted = jaccard_score(true, matched, average='weighted')

    adj_rand = adjusted_rand_score(true, matched)  # between -1 and 1
    
    df_res = pd.DataFrame({'ACC':acc, 'NMI':nmi, 'F1':f1_weighted, 'ARI':adj_rand}, index=[''])
    
    try:
        display(df_res)
    except:
        print(df_res)

#     # confusion matrix
#     confmat = confusion_matrix(true, matched, normalize='true')
#     categories = range(len(confmat)) if categories is None else categories
#     df_cm = pd.DataFrame(confmat, categories, categories)
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}) # font size

#     print("normalized confusion matrix =")
#     plt.show()
    
    return df_res


def balance_class_idx(labels, n_per_class=None):
    
    lset = np.unique(labels)
    num_class = [np.sum(labels==ll) for ll in lset]
    small_num = np.min(num_class)
    n_per_class = small_num if n_per_class is None else n_per_class
    
    lim_lo = min(small_num, n_per_class)

    idx = np.arange(len(labels))
    idx_sample = []
    for li,ll in enumerate(lset):
        bi = labels==ll
        ni = num_class[li]
        idx_s = idx[bi][np.random.RandomState(li).permutation(ni)[:lim_lo]]
        idx_sample.append(idx_s)
        
    return np.hstack(idx_sample)


def tsne_visualize_nparts(X, y, ax=None, title='', seed=42, disp_centers=None, configs=[{}]):
    
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {'doc': ['orange','red','royalblue','green', 'magenta', 'grey'], 'word': ['tan','coral','skyblue','lightgreen', 'plum', 'lightgrey']}
    
    if disp_centers is not None:
        ncls = disp_centers.shape[0]
        x = TSNE(n_components=2, random_state=seed).fit_transform(np.vstack([disp_centers, X]))
        c = x[:ncls]
        x = x[ncls:]
    else:
        x = TSNE(n_components=2, random_state=seed).fit_transform(X)

    # nparts = len(configs)
    for conf in configs:
        idx = np.array(conf['n'])
        alpha = conf['alpha'] if 'alpha' in conf else .2
        symbol = conf['symbol'] if 'symbol' in conf else '.'
        tag = conf['tag'] if 'tag' in conf else ''
        
        y_disp = y[idx]
        x_disp = x[idx]
       
        y_set = sorted(np.unique(y_disp))
        for ki, yi in enumerate(y_set):
            
            if tag=='word' and disp_centers is not None:
                ax.plot(c[ki, 0], c[ki, 1], 'x', label=None, c=colors[tag][ki],
                       markeredgewidth=4, markersize=12)
                
            bi = y_disp==yi
            ax.scatter(x_disp[bi,0], x_disp[bi,1], marker=symbol, label='{}-{}'.format(tag,yi), c=colors[tag][ki], alpha=alpha)
            

                
        ax.set_xlabel('Dimension 1', fontsize=13)
        ax.set_ylabel('Dimension 2', fontsize=13)
    ax.set_title(title)
    ax.legend()
    
    return x
    

def tsne_visualize(X, y, prob_gmm=None, ax=None, seed=42, **kwargs):
    
    alpha = kwargs['alpha'] if 'alpha' in kwargs else .2
    symbol = kwargs['symbol'] if 'symbol' in kwargs else '.'
    tag = kwargs['tag'] if 'tag' in kwargs else ''
    title = kwargs['title'] if 'title' in kwargs else ''
        
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    X_emb = TSNE(n_components=2, random_state=seed).fit_transform(X)

    if ax is None:
        _, ax = plt.subplot(1, 1)

    y_set = sorted(np.unique(y))
    for ki, yi in enumerate(y_set):
        ax.plot(X_emb[y==yi,0], X_emb[y==yi,1], symbol, alpha=alpha, label='{}-{}'.format(tag,yi), color=colors[ki])
        
    if prob_gmm is not None:
        gmm, mapping = prob_gmm # match color        
        k = gmm.means_.shape[0]
        centers = np.zeros((k,2)) 
        colors = np.asarray(colors[:k])[[p[1] for p in sorted(mapping.items())]]

        for i in range(k):
            density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
            centers[i, :] = X_emb[np.argmax(density)]
            
        overlay_gmm_full(X_emb, gmm.weights_, centers, gmm.covariances_, ax=ax, colors=colors)
                
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    ax.set_title(title)
    ax.legend()
    

def make_ellipses(gmm, ax, colors):
    
    n_gaussians = gmm.means_.shape[0]
    for n in range(n_gaussians):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 200. * np.sqrt(2.) * np.sqrt(v)
        ell = patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=colors[n])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
#         ax.add_artist(ell)
#         ax.set_aspect('equal', 'datalim')


def overlay_gmm_full(points, w, mu, covar, ax=None, colors=None):
    '''
    plots points and their corresponding gmm model in 2D
    Input: 
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors
    
#     print(points.shape, w.shape, mu.shape, covar.shape) # (3865, 2) (4,) (2, 4) (4, 4)
    stdev = np.sqrt(np.abs(covar))
    
    n_gaussians = mu.shape[0]
    N = int(np.round(points.shape[0] / n_gaussians))
    
    # Visualize data
    if ax is None:
        _,ax = plt.subplots(1,1)
        ax.set_xlim(min(points[:,0]), max(points[:,0]))
        ax.set_ylim(min(points[:,1]), max(points[:,1]))
            
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        
        stdev_i = stdev.T[i]
        covariances = stdev_i[:2, :2]**2
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        angle = 0 if np.isnan(angle) else angle
        
        for j in list(range(800))[::100]:
            ax.add_patch(
                patches.Ellipse(mu[i,:], width=(j+1) * stdev_i.T[0, i], height=(j+1) *stdev_i.T[1, i],
                                angle = 180+angle, edgecolor='none', facecolor=colors[i], alpha=1.0/(0.5*j/25+1)
                               )
            )
            
            
