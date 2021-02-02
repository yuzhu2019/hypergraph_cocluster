import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from method import clustering
from utils import munkres_assignment, eval_summary, balance_class_idx

# Data Preprocessing

dataset_name = '20newsgroups'
categories = sorted(['comp.os.ms-windows.misc', 'rec.autos', 'sci.crypt', 'talk.politics.guns']) # Dataset 1
# categories = sorted(['alt.atheism', 'comp.graphics', 'misc.forsale', 'rec.sport.hockey', 'sci.electronics', 'talk.politics.mideast']) # Dataset 2
remove = ()
sparsity_b1 = 0.2
sparsity_b2 = 0.002
num_words = 2000
n_per_class = None

# dataset_name = 'rcv1'
# # categories = sorted(['CCAT', 'ECAT', 'GCAT', 'MCAT']) # Dataset 3
# categories = sorted(['C15', 'C18', 'E31', 'E41', 'GCRIM', 'GDIS', 'M11', 'M14']) # Dataset 4
# sparsity_b1 = 0.2
# sparsity_b2 = 0.002
# num_words = 2000
# n_per_class = 1000  # number of documents kept for each class

print('***Load the dataset.')
if dataset_name == '20newsgroups':
    dataset = data.Text20News(subset='all', categories=categories, remove=remove, shuffle=True, random_state=42)
else:  # elif dataset_name == 'rcv1':
    dataset = data.TextRCV1(data_dir='./data/RCV1', subset='all', categories=categories)

print('***Transform text to a-z (lowercase) and (single) whitespace.')
dataset.clean_text(num='substitute')

print('***Count words.')
dataset.vectorize(stop_words='english')

print('***Remove documents containing less than 20 words.')
dataset.remove_short_documents(nwords=20, vocab='full')

print('***Remove documents containing images.')
dataset.remove_encoded_images()

if n_per_class is not None:
    print('***Downsampling.')
    idx_b = balance_class_idx(dataset.labels, n_per_class=n_per_class)
    dataset.keep_documents(idx_b)

print('***Remove words appearing in more than {} percent and less than {} percent documents'.format(sparsity_b1*100, sparsity_b2*100))
dataset.remove_frequent_words(sparsity_b1=sparsity_b1, sparsity_b2=sparsity_b2)

print('***Keep top ' + str(num_words) + ' frequent words.')
dataset.keep_top_words(num_words, 20)

print('***Remove documents containing less than 5 (selected) words.')
dataset.remove_short_documents(nwords=5, vocab='selected')

print('***Compute tf-dif.')
dataset.compute_tfidf()

print('***Construct word labels from class conditional word distribution.')
dataset.class_conditional_word_dist()

dataset.data_info(show_classes=True)

tfidf = dataset.tfidf.astype(np.float32)  # size: (num of documents) x (num of words), scipy.sparse.csr.csr_matrix
d_labels = dataset.labels  # labels of documents
w_labels = dataset.labels_word  # labels of words

k = len(categories)
ndoc = tfidf.shape[0]
nword = tfidf.shape[1]

rs = 42

# Methods

print('star expansion 1 - computing ...')
labels_pred_s1_, features_s1 = clustering(tfidf, k, 'star_expansion', alg='alg1', separate=False, verbose=False, seed=rs)
labels_pred_s1_doc = munkres_assignment(labels_pred_s1_[:ndoc], d_labels)
labels_pred_s1_word = munkres_assignment(labels_pred_s1_[ndoc:], w_labels)

print('star expansion 2 - computing ...')
labels_pred_s2_, features_s2 = clustering(tfidf, k, 'star_expansion', alg='alg2', separate=False, verbose=False, seed=rs)
labels_pred_s2_doc = munkres_assignment(labels_pred_s2_[:ndoc], d_labels)
labels_pred_s2_word = munkres_assignment(labels_pred_s2_[ndoc:], w_labels)

# print('star expansion 1 - dual graph - computing ...')
# labels_pred_ds1_, features_ds1 = clustering(tfidf.T, k, 'star_expansion', alg='alg1', separate=False, verbose=False, seed=rs)
# labels_pred_ds1_doc = munkres_assignment(labels_pred_ds1_[nword:], d_labels)
# labels_pred_ds1_word = munkres_assignment(labels_pred_ds1_[:nword], w_labels)

# print('star expansion 2 - dual graph - computing ...')
# labels_pred_ds2_, features_ds2 = clustering(tfidf.T, k, 'star_expansion', alg='alg2', separate=False, verbose=False, seed=rs)
# labels_pred_ds2_doc = munkres_assignment(labels_pred_ds2_[nword:], d_labels)
# labels_pred_ds2_word = munkres_assignment(labels_pred_ds2_[:nword], w_labels)

print('baseline 1 - raw - computing ...')
labels_pred_raw_, (features_raw_doc, features_raw_word) = clustering(tfidf, k, 'raw', separate=True, verbose=False, seed=rs)
labels_pred_raw_doc = munkres_assignment(labels_pred_raw_[:ndoc], d_labels)
labels_pred_raw_word = munkres_assignment(labels_pred_raw_[ndoc:], w_labels)

print('baseline 2 - bipartite spectral graph partitioning - computing ...')
labels_pred_sbc_, features_sbc = clustering(tfidf, k, 'sbc', separate=False, verbose=False, seed=rs)
labels_pred_sbc_doc = munkres_assignment(labels_pred_sbc_[:ndoc], d_labels)
labels_pred_sbc_word = munkres_assignment(labels_pred_sbc_[ndoc:], w_labels)

print('baseline 3 - clique expansion - computing ...')
labels_pred_clq_, (features_clq_doc, features_clq_word) = clustering(tfidf, k, 'clique_expansion', separate=True, verbose=False, seed=rs)
labels_pred_clq_doc = munkres_assignment(labels_pred_clq_[:ndoc], d_labels)
labels_pred_clq_word = munkres_assignment(labels_pred_clq_[ndoc:], w_labels)

# Performance evaluation

doc_acc = []
doc_keys = []

print('baseline 1 - raw - evaluate doc')
doc_acc.append(eval_summary(labels_pred_raw_doc, d_labels))
doc_keys.append('naive')

print('baseline 2 - bipartite spectral graph partitioning - evaluate doc')
doc_acc.append(eval_summary(labels_pred_sbc_doc, d_labels))
doc_keys.append('bi-spec')

print('baseline 3 - clique expansion - evaluate doc')
doc_acc.append(eval_summary(labels_pred_clq_doc, d_labels))
doc_keys.append('c-spec')

print('star expansion 1 - evaluate doc')
doc_acc.append(eval_summary(labels_pred_s1_doc, d_labels))
doc_keys.append('s-spec-1')

print('star expansion 2 - evaluate doc')
doc_acc.append(eval_summary(labels_pred_s2_doc, d_labels))
doc_keys.append('s-spec-2')

# print('star expansion 1 - dual graph - evaluate doc')
# doc_acc.append(eval_summary(labels_pred_ds1_doc, d_labels))
# doc_keys.append('s-spec-dual-1')

# print('star expansion 2 - dual graph - evaluate doc')
# doc_acc.append(eval_summary(labels_pred_ds2_doc, d_labels))
# doc_keys.append('s-spec-dual-2')

doc_eval = pd.concat(doc_acc, keys=doc_keys)

doc_eval.T.plot(kind='bar', title="clustering accuracy of documents", figsize=(8, 5), rot=0)
plt.legend(loc='best')
plt.xlim([-.5, 3.7])
plt.show()

word_acc = []
word_keys = []

print('baseline 1 - raw - evaluate word')
word_acc.append(eval_summary(labels_pred_raw_word, w_labels))
word_keys.append('naive')

print('baseline 2 - bipartite spectral graph partitioning - evaluate word')
word_acc.append(eval_summary(labels_pred_sbc_word, w_labels))
word_keys.append('bi-spec')

print('baseline 3 - clique expansion - evaluate word')
word_acc.append(eval_summary(labels_pred_clq_word, w_labels))
word_keys.append('c-spec')

print('star expansion 1 - evaluate word')
word_acc.append(eval_summary(labels_pred_s1_word, w_labels))
word_keys.append('s-spec-1')

print('star expansion 2 - evaluate word')
word_acc.append(eval_summary(labels_pred_s2_word, w_labels))
word_keys.append('s-spec-2')

# print('star expansion 1 - dual graph - evaluate word')
# word_acc.append(eval_summary(labels_pred_ds1_word, w_labels))
# word_keys.append('s-spec-dual-1')

# print('star expansion 2 - dual graph - evaluate word')
# word_acc.append(eval_summary(labels_pred_ds2_word, w_labels))
# word_keys.append('s-spec-dual-2')

word_eval = pd.concat(word_acc, keys=word_keys)

word_eval.T.plot(kind='bar', title="clustering accuracy of words", figsize=(8, 5), rot=0)
plt.legend(loc='best')
plt.xlim([-.5, 3.7])
plt.show()
