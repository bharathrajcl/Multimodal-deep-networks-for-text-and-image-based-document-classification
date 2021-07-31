# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 06:50:47 2021

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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import model_build
import preprocessing_code
import json
from PIL import Image,ImageFont,ImageDraw
import pytesseract
import cv2
import numpy as np
from tensorflow.keras.layers import Dense,Flatten,Conv2D
import pickle
#pytesseract.pytesseract.tesseract_cmd = '‪C:/Program Files/Tesseract-OCR/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


with open('training_config_data.json') as data:
    training_config_data = json.load(data)
    
    
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df


def embed_matrix_vector(word_index,embeddings_index,nb_words,embed_dim):
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


def read_short_data(file_path,no_of_rows_per_label):
    df = read_data(file_path)
    
    label_unique = list(set(df.label))
    dic_label = {}
    for i in label_unique:
        dic_label[i] = 0
        
    df_values = df.values
    short_data = []
    for i in df_values:
        for j in dic_label:
            if(dic_label[j] < no_of_rows_per_label):
                short_data.append(i.tolist())
            dic_label[j] += 1
    df = pd.DataFrame(short_data,columns=['file_name','label'])
    return df


path = training_config_data['data_folder_path']
def create_sentence_list(df):
    trainLines = []
    global path
    for c,i in enumerate(df['file_name']):
        if(c%1000 == 0):
            print(c)
        te = open(path+'/'+i+'.txt',encoding='utf-8')
        text = te.read()
        trainLines.append(text)
    return trainLines

def image_read_reshape(img_file):
    global path
    global training_config_data
    img = cv2.imread(path+'/'+img_file+'.jpg')
    #print(type(training_config_data['image_size']),img.shape,type(img))
    img = cv2.resize(img,tuple(training_config_data['image_size']))
    img_shape = img.shape
    img = np.reshape(img,(1,img_shape[0],img_shape[1],3))
    img = img.astype('float64')
    return img

def text_read_encode(text_file,tokenizer,length):
    global path
    text_object = open(path+'/'+text_file+'.txt',encoding='utf-8')
    text = text_object.read()
    return text

def create_tokenizer(lines,MAX_NB_WORDS):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer,lines,length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded,maxlen = length,padding = 'post')
    return padded

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

def data_with_batch(ds_data,index,batch_size,tokenizer,length):
    global encode_text
    global training_config_data
    start_index = 0
    end_index = 0
    if(index == 0):
        start_index = 0
        end_index = start_index+batch_size
    if((index+batch_size) < len(ds_data)):
        start_index = index
        end_index = index+batch_size
    if((index+batch_size) >= len(ds_data)):
        start_index = index
        end_index = len(ds_data)
        
    ds_data_act = ds_data[start_index:end_index]
    images = []
    texts = []
    labels = []
    for each_data in ds_data_act:
        texts.append(text_read_encode(each_data[0], tokenizer, length))
        images.append(image_read_reshape(each_data[0]))
        labels.append(each_data[1])
    if(training_config_data['Embedding_required']):
        texts = encode_text(tokenizer, texts, length)
    else:
        texts = encode_text(tokenizer, texts, length)
    #print(texts.shape,'encoding fun preprocessing')
    images = np.array(images)
    images_shape = images.shape
    
    images = np.reshape(images,(images_shape[0],images_shape[-3],images_shape[-2],images_shape[-1]))
    #labels = np.array(labels)
    #images = tf.convert_to_tensor(images,dtype=tf.float64)
    #texts = tf.convert_to_tensor(texts,dtype=tf.float64)
    
    return images,texts,labels

    
def create_image_file_infer(text):
    act_text = text.split()
    act_new_text = []
    img = Image.new('RGB',(256,256),color = (255,255,255))
    d = ImageDraw.Draw(img)
    for count,each_word in enumerate(act_text):
        if(count != 0 and count%7 == 0):
            act_new_text.append('\n')
        try:
            d.multiline_text((0,100), each_word,fill=(0,0,0))
            act_new_text.append(each_word)
        except:
            continue
    act_new_text = ' '.join(act_new_text)
    img_new = Image.new('RGB',(256,256),color = (255,255,255))
    d_new = ImageDraw.Draw(img_new)
    d_new.multiline_text((0,10), act_new_text,fill=(0,0,0))
    #img_new.save(path+'/'+str(index)+'_'+label+'.jpg')
    img_new = np.array(img_new)
    return img_new 
    
def read_text_file_infer(text_path):
    te = open(text_path)
    text = te.read()
    te.close()
    return text

def read_image_file_infer(image_path):
    img = cv2.imread(image_path)
    return img


def create_text_file_infer(image_data):
    text = pytesseract.image_to_string(image_data)
    return text
    
def load_saved_models(training_config_data):
    model_image = tf.keras.models.load_model(training_config_data['image_model_path'])
    model_text = tf.keras.models.load_model(training_config_data['text_model_path'])
    model_merge = tf.keras.models.load_model(training_config_data['merge_model_path'])
    return model_image,model_text,model_merge