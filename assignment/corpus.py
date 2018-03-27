# -*- coding: utf-8 -*-

#  from pythainlp.tokenize import word_tokenize, dict_word_tokenize
import deepcut
import re

tokens = []

with open('discard.txt', 'r') as df:
    discard = [x.decode('utf-8').strip().lower() for x in df.readlines()]
    with open('corpus/source_list.txt', 'r') as src_list:
        for src in src_list:
            with open('corpus/sources/' + src.rstrip(), 'r') as sf:
                text = ' '.join([x.decode('utf-8').strip().lower() for x in sf.readlines()])
                _tokens = [x.strip().replace(u' ', u'') for x in deepcut.tokenize(text, custom_dict='dict.txt') if x.strip().replace(u' ', u'') != u'']
                tokens += [x for x in _tokens if x not in discard]

corpus = ' '.join(tokens)
corpus = re.sub(r'(\d{1,3}) , (\d{3}) , (\d{3})', r'\1,\2,\3', corpus)
corpus = re.sub(r'(\d{1,3}) , (\d{3})', r'\1,\2', corpus)
corpus = re.sub(r'(\d+) \. (\d+)', r'\1.\2', corpus)
corpus = re.sub(r'(\d+) \.', r'\1.', corpus)
corpus = corpus.replace(u' – ', u'-')
corpus = corpus.replace(u' - ', u'-')
corpus = corpus.replace(u' %', u'%')
print(corpus.encode('utf-8'))
