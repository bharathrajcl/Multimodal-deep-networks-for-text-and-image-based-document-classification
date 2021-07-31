# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:04:26 2021

@author: Bharathraj C L
"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import model_build
import preprocessing_code
import json
import numpy as np
import embedding_matrix_creation

with open('training_config_data.json') as data:
    training_config_data = json.load(data)
    

file_name = training_config_data['base_file_path']
df_master = preprocessing_code.read_data(file_name)
print('master dataset size :',df_master.shape)
no_of_rows_per_label = training_config_data['No_of_rows_per_label']

df = preprocessing_code.read_short_data(file_name,no_of_rows_per_label)
print('considered dataset size : ',df.shape)
trainLines = preprocessing_code.create_sentence_list(df)
MAX_NB_WORDS = 1000
embed_dim = 100 
length = training_config_data['length']

#file_path = 'C:/Users/Bharathraj C L/Downloads/paper to implement/page stream tensorflow/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec'
import pickle

# saving


# loading

if(training_config_data['Embedding_required']):
    tokenizer = preprocessing_code.create_tokenizer(trainLines,length)
    
    embedding_matrix = embedding_matrix_creation.create_embedding(trainLines,length)
else:
    embedding_matrix = None
    tokenizer = preprocessing_code.create_tokenizer(trainLines,length)
    #trainLines = preprocessing_code.encode_text(tokenizer,trainLines,length)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
vocab_size = len(tokenizer.word_index)+1

print('load model started')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
oe = OneHotEncoder()
label = oe.fit_transform(np.array(df['label']).reshape(-1,1)).toarray()
#x_train,x_test,y_train,y_test = train_test_split(df['file_name'],df['label'],random_state = 2000,test_size=0.1)
x_train,x_test,y_train,y_test = train_test_split(df['file_name'],label,random_state = 2000,test_size=0.1)
if(training_config_data['image_model_build_path'] == 'scratch'):
    model_image = model_build.MyModel_image()
else:
    model_image = tf.keras.models.load_model(training_config_data['image_model_path'])
    
if(training_config_data['text_model_build_path'] == 'scratch'):
    model_text = model_build.MyModel_text(vocab_size,length,embedding_matrix)
else:
    model_text = tf.keras.models.load_model(training_config_data['text_model_path'])
    
if(training_config_data['merge_model_build_path'] == 'scratch'):
    model_merge = model_build.MyModel_merge()
else:
    model_merge = tf.keras.models.load_model(training_config_data['merge_model_path'])
''' 
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss_image = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy_image = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss_image = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy_image = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

train_loss_text = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy_text = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss_text = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy_text = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

train_loss_merge = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy_merge = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss_merge = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy_merge = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
'''

loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss_image = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy_image = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss_image = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy_image = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

train_loss_text = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy_text = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss_text = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy_text = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

train_loss_merge = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy_merge = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss_merge = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy_merge = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

train_option_dict = training_config_data['train_option_dict']

@tf.function
def train_step_with_pretrained(images,texts,labels):
    global train_option_dict
    #print(type(images),type(texts),type(labels),'new data')
    with tf.GradientTape() as tape_text:
        #print('in text gradient')
        #print(type(images),type(texts),type(labels),'text')
        predictions_text,intermediate_text_output = model_text([images,texts])
        loss_text = loss_object(labels,predictions_text)
    
    with tf.GradientTape() as tape_image:
        #print('in image gradient')
        #print(type(images),type(texts),type(labels),'image')
        predictions_image,intermediate_image_output = model_image([images,texts])
        #print('image gradient completed')
        loss_image = loss_object(labels,predictions_image)    
    
    with tf.GradientTape() as tape_merge:
        #print('in merge gradient')
        #print(type(intermediate_image_output),type(intermediate_text_output),type(labels),'merge')
        predictions_merge = model_merge([intermediate_image_output,intermediate_text_output])
        loss_merge = loss_object(labels,predictions_merge)
      
    
    gradients_image = tape_image.gradient(loss_image,model_image.trainable_variables)
    optimizer.apply_gradients(zip(gradients_image,model_image.trainable_variables))
    global train_loss_image
    global train_accuracy_image
    train_loss_image_tensor = train_loss_image(loss_image)
    train_accuracy_image_tensor = train_accuracy_image(labels,predictions_image)
    
    gradients_text = tape_text.gradient(loss_text,model_text.trainable_variables)
    optimizer.apply_gradients(zip(gradients_text,model_text.trainable_variables))
    global train_loss_text
    global train_accuracy_text
    train_loss_text_tensor = train_loss_text(loss_text)
    train_accuracy_text_tensor = train_accuracy_text(labels,predictions_text)
    
    
    gradients_merge = tape_merge.gradient(loss_merge,model_merge.trainable_variables)
    optimizer.apply_gradients(zip(gradients_merge,model_merge.trainable_variables))
    global train_loss_merge
    global train_accuracy_merge
    train_loss_merge_tensor = train_loss_merge(loss_merge)
    train_accuracy_merge_tensor = train_accuracy_merge(labels,predictions_merge)
    


@tf.function
def test_step_with_pretrained(images,texts,labels):
    
    predictions_image,intermediate_image_output = model_image((images,texts),training = False)
    loss_image = loss_object(labels,predictions_image)
    
    predictions_text,intermediate_text_output = model_text((images,texts),training = False)
    loss_text = loss_object(labels,predictions_text)
    
    predictions_merge = model_merge((intermediate_image_output,intermediate_text_output),training = False)
    loss_merge = loss_object(labels,predictions_merge)
    
    global test_loss_image
    global test_accuracy_image
    
    test_loss_image_tensor = test_loss_image(loss_image)
    test_accuracy_image_tensor = test_accuracy_image(labels,predictions_image)
    
    global test_loss_text
    global test_accuracy_text
    
    test_loss_text_tensor = test_loss_text(loss_text)
    test_accuracy_text_tensor = test_accuracy_text(labels,predictions_text)
    
    global test_loss_merge
    global test_accuracy_merge
    
    test_loss_merge_tensor = test_loss_merge(loss_merge)
    test_accuracy_merge_tensor = test_accuracy_merge(labels,predictions_merge)
    
    
    
EPOCHS = training_config_data['epochs']
batch_size = training_config_data['batch_size']


def save_all_models(training_config_data):
    global model_image
    global model_text
    global model_merge
    
    train_option_dict = training_config_data['train_option_dict']
    if(train_option_dict['image_model'] == True):
        model_image.save(training_config_data['image_model_path'])
    if(train_option_dict['text_model'] == True):
        model_text.save(training_config_data['text_model_path'])
    if(train_option_dict['merge_model'] == True):
        model_merge.save(training_config_data['merge_model_path'])
        

def main_train(EPOCHS,batch_size,x_train,y_train,x_test,y_test):
    global tokenizer
    global length
    global train_loss_image
    global train_accuracy_image
    
    global train_loss_text
    global train_accuracy_text
    
    global train_loss_merge
    global train_accuracy_merge
    
    global test_loss_image
    global test_accuracy_image
    
    global test_loss_text
    global test_accuracy_text
    
    global test_loss_merge
    global test_accuracy_merge
    
    global model_image
    global model_text
    global model_merge
    
    for epoch in range(EPOCHS):
        train_accuracy_image.reset_states()
        train_loss_image.reset_states()
        test_loss_image.reset_states()
        test_accuracy_image.reset_states()
        
        train_loss_text.reset_states()
        train_accuracy_text.reset_states()
        test_loss_text.reset_states()
        test_accuracy_text.reset_states()
        
        train_accuracy_merge.reset_states()
        train_loss_merge.reset_states()
        test_loss_merge.reset_states()
        test_accuracy_merge.reset_states()
        
        train_ds = list(zip(x_train,y_train))
        test_ds = list(zip(x_test,y_test))
        
        for data_index in range(0,len(train_ds),batch_size):
            images,texts,labels = preprocessing_code.data_with_batch(train_ds,data_index,batch_size,tokenizer,length)
            #print(type(images),type(texts),type(labels),images.shape,'train')
            train_step_with_pretrained(images, texts, labels)
            
        for data_index in range(0,len(test_ds),batch_size):
            test_images,test_texts,test_labels = preprocessing_code.data_with_batch(test_ds,data_index,batch_size,tokenizer,length)
            #print(type(test_images),type(test_texts),type(test_labels),'test')
            test_step_with_pretrained(test_images, test_texts, test_labels)
    #return images,texts,labels
    
        print(
            f'Epoch {epoch+1},','\n',
            #f'Train Loss image: {train_loss_image.result()},'
            f'Train Accuracy image: {train_accuracy_image.result()*100},','\n',
            #f'Test Loss image: {test_loss_image.result()},'
            f'Test Accuracy image: {test_accuracy_image.result()*100},','\n',
            #f'Train Loss text: {train_loss_text.result()},'
            f'Train Accuracy text: {train_accuracy_text.result()*100},','\n',
            #f'Test Loss text: {test_loss_text.result()},'
            f'Test Accuracy text: {test_accuracy_text.result()*100},','\n',
            #f'Train Loss merge: {train_loss_merge.result()},'
            f'Train Accuracy merge: {train_accuracy_merge.result()*100},','\n',
            #f'Test Loss merge: {test_loss_merge.result()},'
            f'Test Accuracy merge: {test_accuracy_merge.result()*100},','\n',
            )
        
    
main_train(EPOCHS,batch_size,x_train,y_train,x_test,y_test)

save_all_models(training_config_data)

