#!/usr/bin/python
# -*- coding: utf-8 -*-

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

labels = []
for f in text_files:
    if f.endswith('.txt'):
        labels.append(f)
#print(labels)

tfidf_vectorizer = TfidfVectorizer()
documents = [open(f, encoding="utf-8").read() for f in text_files if f.endswith('.txt')]
sparse_matrix = tfidf_vectorizer.fit_transform(documents)


pairwise_similarity = sparse_matrix * sparse_matrix.T
pairwise_similarity_array = pairwise_similarity.toarray()

#Clustering
num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(pairwise_similarity_array)

clusters = km.labels_.tolist()
print (clusters)

#dendrograms
X = pairwise_similarity_array
fig = ff.create_dendrogram(X, orientation='bottom', labels=(labels))
fig.update_layout(width=800, height=500)
fig.show()