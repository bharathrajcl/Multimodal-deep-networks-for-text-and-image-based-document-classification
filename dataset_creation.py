# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 13:50:46 2021

@author: Bharathraj C L
"""

from PIL import Image,ImageFont,ImageDraw
import os
import pandas as pd
from nltk.corpus import stopwords
import glob


import json
with open('training_config_data.json') as data:
    training_config_data = json.load(data)
    
path = training_config_data['data_folder_path']
temp_path = './main_dataset/'
data_list = os.listdir('./main_dataset/')
act_data_list = [x for x in data_list if(x.split('.')[-1] == 'tsv')]

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''




def remove_punct(each_token):
    no_punct = ""
    for char in each_token:
        if char not in punctuations:
            no_punct = no_punct+char
        else:
            no_punct = no_punct+' '
    return no_punct

stop_words = set(stopwords.words('english'))
def clean_doc(doc):
    tokens = doc.split()
    tokens = [remove_punct(word) for word in tokens]
    tokens = ' '.join(tokens).split(' ')
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [i for i in tokens if not i.isdigit()]
    tokens = [word for word in tokens if len(word) > 2]
    tokens = ' '.join(tokens)
    return tokens


def create_image_file(text,label,path,index):
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
    img_new.save(path+'/'+str(index)+'_'+label+'.jpg')
    
def create_text_file(text,label,path,index):
    te = open(path+'/'+str(index)+'_'+label+'.txt','w+',encoding = 'utf-8')
    te.write(text)
    te.close()
    


def main_method(data_list,count_per_label):
    global path
    for count,i in enumerate(data_list):
        print(count,'outer')
        #print(i)
        label_data = i.split('.')[0]
        #print(label_data)
        if label_data not in []:
            df = pd.read_csv(temp_path+i,error_bad_lines=False,sep='\t')
            df = df.dropna()
            r_head = pd.DataFrame(df['review_headline'],columns =['review_headline'])
            r_body = pd.DataFrame(df['review_body'],columns = ['review_body'])
            del df
            all_data = pd.concat([r_head,r_body],axis = 1)
            all_data = all_data.values
            del r_head
            del r_body
            print(all_data.shape)
            data_result_100 = []
            count = 0
            index = 0
            for each_data in all_data:
                #print(each_data)
                count = count+1
                text= each_data[0]+' '+each_data[1]
                pre_text = clean_doc(text).split(' ')
                if(len(pre_text) > 100):
                    pre_text = ' '.join(pre_text)
                    create_image_file(pre_text, label_data, path, index)
                    create_text_file(pre_text, label_data, path, index)
                    index = index+1
                if(index > count_per_label):
                    break
                if(count%1000 == 0):
                    print(count, index)
                

            
main_method(act_data_list, 5000)
    

def create_relevant_csv(file_path,output_path):
    data = os.listdir(file_path)
    act_data = []
    
    for c,i in enumerate(data):
        temp = i.split('.')[0]
        name,label = temp,temp.split('_')[-1]
        new_temp = name+';'+label
        act_data.append(new_temp)
        
    act_data = list(set(act_data))
    new_act_data = []
    for i in act_data:
        new_temp = i.split(';')
        new_act_data.append(new_temp)
    df = pd.DataFrame(new_act_data,columns=['file_name','label'])
    df.to_csv(output_path,index = False)

create_relevant_csv(training_config_data['data_folder_path'], training_config_data['base_file_path'])


    
'''
new_ui = {
"data_folder_path":"C:/Users/Bharathraj C L/Downloads/paper to implement/page stream tensorflow/merge_data",
"image_size":[128,128],
"base_file_path":"C:/Users/Bharathraj C L/Downloads/paper to implement/page stream tensorflow/data_all.csv",
"No_of_rows_per_label":100,
"Embedding_required" : True,
"train_option_dict": {"image_model": True,"text_model":True,"merge_model":True},
"epochs":1,
"batch_size":64,
"image_model_path":"save_image_model",
"text_model_path":"save_text_model",
"merge_model_path":"save_merge_model",
"image_model_build_path":"non scratch",
"text_model_build_path":"non scratch",
"merge_model_build_path":"non scratch",
"length":1000
}
import json
with open('training_config_data.json','w+') as data:
    json.dump(new_ui,data)
'''
