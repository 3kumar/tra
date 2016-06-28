# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:46:46 2016

This scripts reads the trained word2vec model(i.e trained on wikipedia data as well as our TRA data)
and generate a dictionaray only for words which are available in out data file.
As the word2vec model have lot of words because it was trained on wikipedia dictionary. The whole model
was not required for our testing.

The output data dict will be used to get the word2vec representation while training TRA model. This script is to
be run all the time whenever data is changed

@author: fox
"""
try:
    import cPickle as p
except:
    import pickle as p
from tra_model import ThematicRoleModel
from gensim.models.word2vec import Word2Vec

W2V_WIKI_TRA_MODEL_FILE='word2vec/sgns-100-tra.model'
OUT_DATA_DICT='data/corpus-word-vectors-100dim.pkl'

sentences_45,labels_45=ThematicRoleModel.load_corpus(corpus_size=45,subset=range(0,45))
sentences_462,labels_462=ThematicRoleModel.load_corpus(corpus_size=462,subset=range(0,462))
sentences=sentences_45+sentences_462

print 'Loading Pre-trained Word2Vec Model...'
model=Word2Vec.load(W2V_WIKI_TRA_MODEL_FILE)
print 'Pre-trained Word2Vec Model Loaded successfully...'

vocab = dict()
for sentence_no, sentence in enumerate(sentences):
    for word in sentence:
         if not vocab.has_key(word):
             vocab[word]=model[word]

with open(OUT_DATA_DICT,'w') as f:
        print 'Dumping file using pickle:',OUT_DATA_DICT
        p.dump(vocab,f)
        print 'Dumped Successfully.'

