# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:13:58 2016


This scripts loads word2vec model trained on wikipedia data  and train the model with our tra data.

Note: Note that the sentences are presented multiplied by 50, to increase the contribution of data in
word2vec representation. This is suggested by the author of gensim.

@author: fox
"""

from tra_model_leak_rate import ThematicRoleModel
from gensim.models.word2vec import Word2Vec

W2V_WIKI_MODEL_FILE='/home/fox/Desktop/word2vec/models/word2vec-sgns/sgns-100.model'
W2V_WIKI_TRA_MODEL_FILE='word2vec/sgns-100-tra.model'

def train_model(model,sentences,isReplace=False):
    # kind of cheat to increase the data to have higher impact of it on vector representation
    print 'Training the model on new sentences...'
    sentences=50 * sentences
    model.train(sentences,total_examples=len(sentences))
    print 'Word2vec model trained successfully on new sentences...'
    normalize_and_save(model,isReplace)

def normalize_and_save(model,isReplace=False):
    print 'Normalizing word vector and saving...'
    if isReplace:
        model.init_sims(replace=True)
    else:
        model.init_sims(replace=False)

    model.save(W2V_WIKI_TRA_MODEL_FILE)
    print 'Model saved successfully...'

sentences_45,labels_45=ThematicRoleModel.load_corpus(corpus_size=45,subset=range(0,45))
sentences_462,labels_462=ThematicRoleModel.load_corpus(corpus_size=462,subset=range(0,462))
sentences=sentences_45 + sentences_462

print 'Loading Pre-trained Word2Vec Model...'
model=Word2Vec.load(W2V_WIKI_MODEL_FILE)
model.build_vocab(sentences,update=True)
print 'Pre-trained Word2Vec Model Loaded successfully...'
model.min_count=1
train_model(model,sentences,False)


