#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib.request import Request, urlopen
import html2text
from bs4 import BeautifulSoup
import re

req  = Request('https://www.zougla.gr/money/agores/article/makron-ke-merkel-simfonisan-tin-prika-tou-tamiou-anakampsis-sta-500-dis-evro', headers={'User-Agent': 'Mozilla/5.0'})
html = urlopen(req).read()
#print (html)

#soup = BeautifulSoup(urllib.request('cnn.gr/news/kosmos/story/220079/pos-i-merkel-kai-o-makron-symfonisan-gia-to-tameio-anakampsis'))
soup = BeautifulSoup(html, features='html.parser')
page = soup.find_all('p')
page = soup.getText()
#print (page)


with open('zougla_extract.txt', 'w', encoding="utf8") as outfile:
    outfile.write('\n'.join([i for i in page.split('\n') if len(i) > 0]))