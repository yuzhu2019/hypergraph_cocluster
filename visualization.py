import data
from method import Cluster
from utils import *

dataset_name = '20newsgroups'
categories = sorted(['comp.os.ms-windows.misc', 'rec.autos', 'sci.crypt', 'talk.politics.guns'])
remove = ()
sparsity_b1 = 0.2
sparsity_b2 = 0.002
num_words = 2000
n_per_class = None

# Data Preprocessing
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
dataset.class_conditional_word_dist(Mprint=20)

dataset.data_info(show_classes=True)

tfidf = dataset.tfidf.astype(np.float32)  # size: (num of documents) x (num of words)
d_labels = dataset.labels  # labels of documents
w_labels = dataset.labels_word  # labels of words

print(np.shape(tfidf))

k = len(categories)
ndoc = tfidf.shape[0]
nword = tfidf.shape[1]

rs = 42

# TSNE
star_cluster = Cluster('star_expansion_alg2', k, separate=False, verbose=False)
labels_pred_ = star_cluster.fit_predict(tfidf)
features_ = star_cluster.features[0]

labels_pred_doc = munkres_assignment(labels_pred_[:ndoc], d_labels)
labels_pred_word = munkres_assignment(labels_pred_[ndoc:], w_labels)

eval_summary(labels_pred_doc, d_labels)
eval_summary(labels_pred_word, w_labels)

fig, ax = plt.subplots(1, 1, figsize=(7,5))
plot_configs = [{'n':range(ndoc), 'symbol':'+', 'tag':'doc', 'alpha':1},
                {'n':range(ndoc,ndoc+nword), 'symbol':'o', 'tag':'word', 'alpha':0.4}]
tsne_visualize_nparts(features_, np.hstack((d_labels, w_labels)), ax=ax, seed=42, configs=plot_configs)
plt.show()

# WordCloud
word_importance = word_freq_inclass(dataset.data, np.hstack((labels_pred_doc, labels_pred_word)))

fig, ax = plt.subplots(1, len(categories), figsize=(15,5))
show_word_cloud(dataset.vocab, labels_pred_word, word_importance, categories, ax.flatten(), seed=0)
plt.tight_layout()
plt.show()
