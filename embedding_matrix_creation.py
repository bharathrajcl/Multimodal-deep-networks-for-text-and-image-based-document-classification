# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 12:31:17 2021

@author: Bharathraj C L
"""

import pandas as pd
import tensorflow as tf
import numpy
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from tqdm import tqdm
import os, re, csv, math, codecs



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
#import model_build
import preprocessing_code
import json
import numpy as np

'''
with open('training_config_data.json') as data:
    training_config_data = json.load(data)
    

file_name = training_config_data['base_file_path']
df_master = preprocessing_code.read_data(file_name)
print('master dataset size :',df_master.shape)
No_of_rows_per_label_for_embeddings_creation = training_config_data['No_of_rows_per_label_for_embeddings_creation']

df = preprocessing_code.read_short_data(file_name,No_of_rows_per_label_for_embeddings_creation)
#print('considered dataset size : ',df.shape)
print('1')
import time
start = time.time()
trainLines = preprocessing_code.create_sentence_list(df)
end = time.time()
print(end-start)
print('2')
MAX_NB_WORDS = 10000
embed_dim = 300
length = training_config_data['length']
'''
def create_embedding(trainLines,length):
    embed_dim = 300
    file_path = 'C:/Users/Bharathraj C L/Downloads/paper to implement/page stream tensorflow/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec'
    tokenizer = preprocessing_code.create_tokenizer(trainLines,length)
    word_index = tokenizer.word_index
    nb_words = min(length, len(word_index))
    embeddings_index = preprocessing_code.extract_embd_file(file_path)
    embedding_matrix,words_not_found = preprocessing_code.embed_matrix_vector(word_index,embeddings_index,nb_words,embed_dim)
    np.save('embedding_matrix.npy', embedding_matrix)
    return embedding_matrix
