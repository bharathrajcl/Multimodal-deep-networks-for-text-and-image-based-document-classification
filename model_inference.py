# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 19:12:48 2021

@author: Bharathraj C L
"""


import tensorflow as tf
import preprocessing_code
import json
import pytesseract
import cv2
import numpy as np
import pickle
#pytesseract.pytesseract.tesseract_cmd = '‪C:/Program Files/Tesseract-OCR/tesseract'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
with open('training_config_data.json') as data:
    training_config_data = json.load(data)


model_image,model_text,model_merge = preprocessing_code.load_saved_models(training_config_data)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

from statistics import mode

def inference_single_data(image_path=None,text_path=None):
    global model_image
    global model_text
    global model_merge
    global training_data_config
    global tokenizer
    if(image_path != None and text_path != None):
        image_data = preprocessing_code.read_image_file_infer(image_path)
        text_data = preprocessing_code.read_text_file_infer(text_path)
        
    if(image_path == None and text_path != None):
        text_data = preprocessing_code.read_text_file_infer(text_path)
        image_data = preprocessing_code.create_image_file_infer(text_data)
        
    if(image_path != None and text_path == None):
        image_data = preprocessing_code.read_image_file_infer(image_path)
        text_data = preprocessing_code.create_text_file_infer(image_data)
    if(image_path == None and text_path == None):
        raise Exception('Please give either Image path or text path')
    
    image_shape_act = training_config_data['image_size']

    image_data = cv2.resize(image_data,tuple(image_shape_act))
    
    
    image_data_shape = image_data.shape
    image_data = np.reshape(image_data,(1,image_shape_act[0],image_shape_act[1],image_data_shape[-1]))
    
    
    length = training_config_data['length']
    text_data = preprocessing_code.encode_text(tokenizer,[text_data],length)
    
    image_data = image_data.astype(np.float32)
    text_data = text_data.astype(np.int32)
    predictions_text,intermediate_text_output = model_text([image_data,text_data])
    predictions_image,intermediate_image_output = model_image([image_data,text_data])
    predictions_merge = model_merge([intermediate_image_output,intermediate_text_output])
    
    temp_result = [predictions_image.numpy()[0],predictions_text.numpy()[0],predictions_merge.numpy()[0]]
    final_result = []
    for i in temp_result:
        final_result.append(tf.math.argmax(i).numpy())
    
    return final_result,temp_result


image_path = './merge_data/0_Kitchen.jpg' 
text_path = './merge_data/0_.txt'
final_result = inference_single_data(None,text_path) 
