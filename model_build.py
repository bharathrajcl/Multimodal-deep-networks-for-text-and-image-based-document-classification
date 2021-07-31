# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 18:24:53 2021

@author: Bharathraj C L
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dropout,Embedding
from tensorflow.keras.layers import Conv1D,LSTM,MaxPooling1D,concatenate

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import MobileNetV2

import json
with open('training_config_data.json') as data:
    training_config_data = json.load(data)

class MyModel_image_pretrained(Model):
    def __init__(self):
        super(MyModel_image_pretrained,self).__init__()
        self.mobilenet = vgg16(weights='imagenet',include_top=False,input_shape=(128,128,3))
        self.flatten = Flatten()
        self.d1 = Dense(128,activation='relu')
        self.d2 = Dense(4)
        
    def call(self,x_image,x_text):
        x = self.mobilenet(x_image)
        x1 = self.flatten(x)
        x = self.d1(x1)
        return self.d2(x),x1

class MyModel_text_pretrained(Model):
    def __init__(self,vocab_size,length,embedding_matrix):
        super(MyModel_text_pretrained,self).__init__()
        self.input1 = Input(shape=(length,))
        if(training_config_data['Embedding_required']):
            self.embd = Embedding(vocab_size,100,weights = [embedding_matrix])
        else:
            self.embd = Embedding(vocab_size,100)
        self.conv1 = Conv1D(64,3,activation = 'relu')
        self.lstm = LSTM(32,return_sequences = True,return_state = True)
        self.flatten = Flatten()
        self.d1 = Dense(128,activation='relu')
        self.d2 = Dense(4)
        
    def call(self,x_image,x_text):
        x = self.embd(x_text)
        x,final_memory_state,final_carry_state = self.lstm(x)
        x = self.conv1(x)
        x1 = self.flatten(x)
        x = self.d1(x1)
        return self.d2(x),x1
    
class MyModel_image(Model):
    def __init__(self):
        super(MyModel_image,self).__init__()
        self.conv2 = Conv2D(32,3,activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128,activation = 'relu')
        self.d2 = Dense(4,activation = 'softmax')
        
    def call(self,x):
        x_image,x_text = x[0],x[1]
        #print(x_image.shape,x_text.shape)
        #print(type(x_image),type(x_text))
        x = self.conv2(x_image)
        x1 = self.flatten(x)
        x = self.d1(x1)
        return self.d2(x),x1
        
    
class MyModel_text(Model):
    def __init__(self,vocab_size,length,embedding_matrix):
        super(MyModel_text,self).__init__()
        self.input1 = Input(shape=(length,))
        if(training_config_data['Embedding_required']):
            self.embd = Embedding(vocab_size,300,weights = [embedding_matrix])
        else:
            self.embd = Embedding(vocab_size,100)        
        self.conv1 = Conv1D(32,3,activation = 'relu')
        self.flatten = Flatten()
        self.d1 = Dense(128,activation = 'relu')
        self.d2 = Dense(4,activation='softmax')
        
    def call(self,x):
        x_image,x_text = x[0],x[1]
        x = self.embd(x_text)
        x = self.conv1(x)
        x1 = self.flatten(x)
        x = self.d1(x1)
        return self.d2(x),x1

class MyModel_merge(Model):
    def __init__(self):
        super(MyModel_merge,self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(32,activation='relu')
        self.d2 = Dense(64,activation='relu')
        self.d3 = Dense(128,activation='relu')
        self.d4 = Dense(4,activation = 'softmax')
        
    def call(self,x):
        x_image,x_text = x[0],x[1]
        x_image = self.flatten(x_image)
        x_text = self.flatten(x_text)
        #print(x_image.shape,x_text.shape)
        x = tf.concat([x_image,x_text],1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)