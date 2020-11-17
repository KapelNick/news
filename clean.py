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


with open('zougla_terms.txt', encoding="utf8") as infile:
    with open('zougla.txt', 'w', encoding="utf8") as outfile:
        for line in infile:
            if "Vb" in line:
                line = line.split()[0]
                outfile.write(line.lower())
                outfile.write('\n')
            if "No" in line:
                line = line.split()[0]
                outfile.write(line.lower())
                outfile.write("\n")
            if "Aj" in line:
                line = line.split()[0]
                outfile.write(line.lower())
                outfile.write("\n")


#VerbGram = r""" Chunk: {<tVb.?>*} """
#NounGram = r""" Chunk: {<tNo.?>*} """
#AjectiveGram = r""" Chunk: {<tAj.?>*} """