#!/usr/bin/python
# -*- coding: utf-8 -*-

from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import os
import codecs
import string, re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet as wn
import plotly.figure_factory as ff
import numpy as np
from sklearn.cluster import KMeans
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path

path = "C:\\Users\\kapel\\Desktop\\texts\\dataset"
text_files = os.listdir(path)

doc_labels = []
for f in text_files:
    if f.endswith('.txt'):
        doc_labels.append(f)
#print(labels)

tfidf_vectorizer = TfidfVectorizer()
documents = [open(f, encoding="utf-8").read() for f in text_files if f.endswith('.txt')]
sparse_matrix = tfidf_vectorizer.fit_transform(documents)


#Similarity
pairwise_similarity = sparse_matrix * sparse_matrix.T
pairwise_similarity_array = pairwise_similarity.toarray()

# load the karate club graph
G = nx.from_numpy_matrix(pairwise_similarity_array)

# compute the best partition
partition = community_louvain.best_partition(G)
#print(partition)
modularity = community_louvain.modularity(partition, G)
print(modularity)

# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('coolwarm', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=100,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
plt.show()

dendro = community_louvain.generate_dendrogram(G)