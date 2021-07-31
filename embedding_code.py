# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 05:58:03 2021

@author: Bharathraj C L
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs


kl = ['my name is bharath',
      'my name is chandan',
      'my name is hema',
      'my name is lakshminarayanaswamy']

MAX_NB_WORDS = 100000
print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(kl)  #leaky
word_seq_train = tokenizer.texts_to_sequences(kl)
#word_seq_test = tokenizer.texts_to_sequences(kl)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))


file_path = 'C:/Users/Bharathraj C L/Downloads/paper to implement/page stream tensorflow/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec'
#load embeddings
def extract_embd_file(file_path):
    embeddings_index = {}
    f = codecs.open(file_path, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    return embeddings_index

#print('found %s word vectors' % len(embeddings_index))#load embeddings

#training params
batch_size = 256 
num_epochs = 8 

#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4

#embedding matrix
print('preparing embedding matrix...')

nb_words = min(MAX_NB_WORDS, len(word_index))
def embed_matrix_vector(word_index,embeddings_index,nb_words):
    words_not_found = []
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    return embedding_matrix,words_not_found


#print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))






'''
print("training CNN ...")
model = Sequential()
model.add(Embedding(nb_words, embed_dim,weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
'''