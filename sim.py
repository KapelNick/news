#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict

#Create list of documents to work with
path = "C:\\Users\\kapel\\Desktop\\data\\dataset"
text_files = os.listdir(path)
#print (text_files)

#check we are only working with text files
documents = [open(f, encoding="utf-8").read() for f in text_files if f.endswith('.txt')]

#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer followed by TfidfTransformer.
tfidf_vectorizer = TfidfVectorizer()

sparse_matrix = tfidf_vectorizer.fit_transform(documents)
#print(sparse_matrix)
#print(tfidf_vectorizer.vocabulary_)


#create a list of labels to use later in the plot
labels = []
for f in text_files:
    if f.endswith('.txt'):
        labels.append(f)
#print(labels)
#print(labels.index('capital_chunks.txt'))


#Optional: Convert Sparse Matrix to Pandas Dataframe for word frequencies
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=tfidf_vectorizer.get_feature_names(),
                  index=documents)
df
#df.to_csv (r'C:\\Users\\kapel\\Desktop\\data\\results\\doc_term_matrix.csv', index = True, header=True, encoding='utf-8')


#Similarity 1st implementation
similarity = cosine_similarity(df, df)
#print(similarity[0])

#Similarity 2nd implementation
pairwise_similarity = sparse_matrix * sparse_matrix.T
sm = pairwise_similarity.toarray()
#print(sm[0])

def argsort_sim_mat(sm):
    idx = [np.argmax(np.sum(sm, axis=1))]
    for i in range(1, len(sm)):
        sm_i = sm[idx[-1]].copy()
        sm_i[idx] = -1
        idx.append(np.argmax(sm_i))
    return np.array(idx)

idx = argsort_sim_mat(sm)
#idx2 = argsort_sim_mat(sim_mat2)
sim_mat_sorted = sm[idx, :][:, idx]
#sim_mat_sorted2 = sim_mat[idx2, :][:, idx2]

#write similarity to file
with open('C:\\Users\\kapel\\Desktop\\data\\results\\doc_term_matrix.xls', 'w') as f:
    for item in doc_term_matrix:
        f.write("%s\n" % item)
        f.write('\n')

fig, ax = plt.subplots(figsize=(20,20))
cax = ax.matshow(sim_mat_sorted, interpolation='spline16')
ax.grid(True)
plt.title('News articles similarity matrix')
plt.xticks(range(24), labels, rotation=90);
plt.yticks(range(24), labels);
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#plt.show()