## Co-clustering Vertices and Hyperedges via Spectral Hypergraph Partitioning

The code for the paper "Co-clustering Vertices and Hyperedges via Spectral Hypergraph Partitioning" by Yu Zhu, Boning Li and Santiago Segarra.

#### Datasets

We adopt two datasets: 20 Newsgroups and Reuters Corpus Volume 1 (RCV1). 

The RCV1 dataset is downloaded from the link: 

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm

In particular, the following ones are needed:

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz

http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz

The code of preprocessing these datasets can be found in data.py

#### Methods

The code of the propsed methods and baseline methods can be found in method.py (and the files under the folder algs)

#### To reproduce the results in our paper

The code for Figure 1: main.py

The code for Figures 2 and 3: visualization.py

