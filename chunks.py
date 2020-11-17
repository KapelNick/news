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

#treetagger tagset explanation http://nlp.ilsp.gr/nlp/tagset_examples/tagset_en/


with open('zougla_terms.txt', 'w', encoding="utf8") as outfile:
    with open('zougla_tag.txt', encoding="utf8") as infile:
        for line in infile:
            if "Vb" in line:
                outfile.write(line)
            if "No" in line:
                outfile.write(line)
            if "Aj" in line:
                outfile.write(line)


#VerbGram = r""" Chunk: {<tVb.?>*} """
#NounGram = r""" Chunk: {<tNo.?>*} """
#AjectiveGram = r""" Chunk: {<tAj.?>*} """