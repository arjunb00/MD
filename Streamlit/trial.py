import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import pandas as pd  
import tensorflow as tf
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.applications import resnet50
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import numpy as np
from pandas_datareader import data
from datetime import datetime, timedelta

def image_process(im):
    width, height = im.size
    new_dimensions = 128
    a = (width - new_dimensions) //2
    b = (height - new_dimensions) //2
    c = (width + new_dimensions) //2
    d = (height + new_dimensions) //2 

    im_tmp = im.crop((a,b,c,d))
    im_tmp = im_tmp.save('new_image.png')

    img=tf.keras.utils.load_img(
                                'new_image.png',
                                grayscale=False,
                                color_mode='rgb',
                                target_size= (128,128),
                                interpolation='nearest',
                            )

    img= tf.keras.utils.img_to_array(img)
    img=resnet50.preprocess_input(img)
    imArr = [img]
    imArr = np.array(imArr)
    print(imArr.shape)
    return fE.predict(imArr)


def df_process():
    entry = []
    entry.append(age)
    if itch == 'Yes':
        entry.append(0)
        entry.append(1)
        entry.append(0)
    elif itch == 'No':
        entry.append(1)
        entry.append(0)
        entry.append(0)
    else:
        entry.append(0)
        entry.append(0)
        entry.append(1)
    
    if grew == 'Yes':
        entry.append(0)
        entry.append(1)
        entry.append(0)
    elif grew == 'No':
        entry.append(1)
        entry.append(0)
        entry.append(0)
    else:
        entry.append(0)
        entry.append(0)
        entry.append(1)

    if hurt == 'Yes':
        entry.append(0)
        entry.append(1)
        entry.append(0)
    elif hurt == 'No':
        entry.append(1)
        entry.append(0)
        entry.append(0)
    else:
        entry.append(0)
        entry.append(0)
        entry.append(1)

    if changed == 'Yes':
        entry.append(0)
        entry.append(1)
        entry.append(0)
    elif changed == 'No':
        entry.append(1)
        entry.append(0)
        entry.append(0)
    else:
        entry.append(0)
        entry.append(0)
        entry.append(1)
    
    if bleed == 'Yes':
        entry.append(0)
        entry.append(1)
        entry.append(0)
    elif bleed == 'No':
        entry.append(1)
        entry.append(0)
        entry.append(0)
    else:
        entry.append(0)
        entry.append(0)
        entry.append(1)

    if region == "ABDOMEN":
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "ARM":
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "BACK":
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "CHEST":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "EAR":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "FACE":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "FOOT":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "FOREARM":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "HAND":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "LIP":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "NECK":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
        entry.append(0)
    elif region == "NOSE":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append(0)
    elif region == "SCALP":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)
        entry.append(0)
        entry.append()
    elif region == "THIGH":
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(0)
        entry.append(1)

    if skin_cancer == 'Yes':
        entry.append(0)
        entry.append(1)
    elif skin_cancer == 'No':
        entry.append(1)
        entry.append(0)

    if cancer == 'Yes':
        entry.append(0)
        entry.append(1)
    elif cancer == 'No':
        entry.append(1)
        entry.append(0)

    if gender == 'Male':
        entry.append(0)
        entry.append(1)
    elif gender == 'Female':
        entry.append(1)
        entry.append(0)

    if smoke == 'Yes':
        entry.append(0)
        entry.append(1)
    elif smoke == 'No':
        entry.append(1)
        entry.append(0)

    if drink == 'Yes':
        entry.append(0)
        entry.append(1)
    elif drink == 'No':
        entry.append(1)
        entry.append(0)

    if elevation == 'Yes':
        entry.append(0)
        entry.append(1)
        entry.append(0)
    elif elevation == 'No':
        entry.append(1)
        entry.append(0)
        entry.append(0)
    else:
        entry.append(0)
        entry.append(0)
        entry.append(1)
    return pd.DataFrame([entry], columns=column_names)


image_input = Input((128, 128, 3))
model = resnet50.ResNet50(include_top=False,input_shape=(128, 128, 3),weights = 'imagenet',pooling='avg')
flattened = tf.keras.layers.Flatten()(model.output)
fc1 = tf.keras.layers.Dense(128, activation='relu', name="AddedDense1")(flattened)
fc2 = tf.keras.layers.Dense(64, activation='softmax', name="AddedDense2")(fc1)
fE = tf.keras.models.Model(inputs=model.input, outputs=fc2)
dL = tf.keras.models.load_model('finalmodel.h5')


st.title('Skin Disease Detection Using Smartphone Images')
st.subheader("Enter your details below")

column_names = ['age', 'itch_False', 'itch_True', 'itch_UNK', 'grew_False', 'grew_True',
       'grew_UNK', 'hurt_False', 'hurt_True', 'hurt_UNK', 'changed_False',
       'changed_True', 'changed_UNK', 'bleed_False', 'bleed_True', 'bleed_UNK',
       'region_ABDOMEN', 'region_ARM', 'region_BACK', 'region_CHEST',
       'region_EAR', 'region_FACE', 'region_FOOT', 'region_FOREARM',
       'region_HAND', 'region_LIP', 'region_NECK', 'region_NOSE',
       'region_SCALP', 'region_THIGH', 'skin_cancer_history_False',
       'skin_cancer_history_True', 'cancer_history_False',
       'cancer_history_True', 'gender_FEMALE', 'gender_MALE', 'smoke_False',
       'smoke_True', 'drink_False', 'drink_True', 'elevation_False',
       'elevation_True', 'elevation_UNK']

disease_names = ['Nevus', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Melanoma']

df = pd.DataFrame(columns=column_names)

with st.form("form1", clear_on_submit = True):

    col1, col2, col3 = st.columns(3)
    gender = col1.selectbox("Enter your gender",["Male", "Female"])
    age = col2.number_input("Enter your age", step = 1, min_value = 1)  
    itch = col3.selectbox("Does the lesion itch?",["Yes", "No", "I don't know"])
    grew = col1.selectbox("Has the lesion grown over a period of time?",["Yes","No", "I don't know"])
    skin_cancer = col2.selectbox("Do your family have a history of skin cancer?",["Yes","No"])
    cancer = col3.selectbox("Do your family have a history of cancer?",["Yes","No"])
    hurt = col3.selectbox("Does the lesion hurt?",["Yes","No", "I don't know"])
    changed = col1.selectbox("Has the lesion changed over time?",["Yes","No", "I don't know"])
    bleed = col2.selectbox("Does the lesion bleed?",["Yes","No", "I don't know"])
    elevation = col2.selectbox("Is the lesion elevated from your skin?",["Yes","No", "I don't know"])
    region = col1.selectbox("What region is the lesion located on?",["ABDOMEN", 'ARM', 'BACK', 'CHEST','EAR', 'FACE', 'FOOT', 'FOREARM','HAND', 'LIP', 'NECK', 'NOSE','SCALP', 'THIGH'])
    smoke = col1.selectbox("Do you smoke?",["Yes","No"])
    drink = col3.selectbox("Do you drink?",["Yes","No"])
    file_up = st.file_uploader("Upload an image of the skin lesion", type="png")   

    
    if file_up != None:
            image = Image.open(file_up)
            dimensions = image.size
            transform = transforms.Compose([
            transforms.CenterCrop(min(dimensions)),
            transforms.Resize(128),])


            st.image(transform(image), caption='Uploaded Image.', use_column_width=True)
    
    if st.form_submit_button("Submit"):
        
        newrow = df_process()
        if file_up != None:
            im = Image.open(file_up)
            features = image_process(im)
            predictions = dL.predict([features,newrow])
            predictions = predictions[0]
            predictions = predictions * 100
            print(predictions)
            disease = disease_names[np.argmax(predictions)]
            print(disease)
            st.write('Your Lesion has a: ')
            for i in range(6):
                st.write(str(predictions[i]) + '% chance of being ' +  disease_names[i])

            cancer = predictions[5] + predictions [1] + predictions[4]
            cancerchance = (cancer / sum(predictions)) * 100
            
            cancerchance = round(cancerchance, 2)
            st.write('')
            if cancerchance > 49:
                st.write('We recommend you see a specialist as you have a ' + str(cancerchance) + ' chance of having skin cancer')
            else:
                st.write('You have a ' + str(cancerchance) + '% chance of having skin cancer, however if you still feel uncertain about your mole, please go see a specialist')


            disease_names = ['Nevus', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Melanoma']
            


    
        
    



                















