#!/usr/bin/env python
# coding: utf-8

# In[1]:

pip install tensorflow


import streamlit as st
import json
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from urllib.request import urlopen
import pandas as pd
import cv2
import math
import io
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

st.set_page_config(layout='wide')


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://raw.githubusercontent.com/adhok/SeeFood/main/luxury-ornamental-mandala-design-background_1159-6794.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


st.sidebar.write("""
                 **Unmasking Authentic Reactions:**

                 Leveraging Facial Emotion Recognition to Reveal Genuine Gastronomic Sentiments Amidst Normative Social Influence
                 """)

st.sidebar.write("""
                 **Northwestern University Computer Vision Final Project**

                    Archit Argawal
                 
                    Winnie Wu
                 
                    Danica Bellchambers
                 
                    Faris Raza
                 """)
# Put data
# details

# Load pre-trained model (MobileNetV2)
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

@st.cache_resource
def load_class_names(class_url):
    with urlopen(class_url) as url:
        class_labels = json.loads(url.read().decode())
    class_names = [class_labels[str(i)][1] for i in range(len(class_labels))]
    return class_names

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def makes_preditions(uploaded_file,preds):
    preprocessed_image = load_and_preprocess_image(uploaded_file)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = decode_predictions(predictions, top=preds)[0]
    return decoded_predictions

def decode_and_print(decoded_predictions):
    # for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    #                 st.write(f"{label} ({score*100:.2f}%)")
    df = pd.DataFrame(decoded_predictions,columns=['ImageNetID','Prediction','Similarity Score'])
    df['Similarity Score'] = df['Similarity Score'].map('{:.2%}'.format)
    st.dataframe(df)

def load_analyse_video(videoFile):
    video_output = []
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            img_array = tf_image.img_to_array(frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=1)[0]
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                video_output.append([label,score*100])
    cap.release()
    return video_output

def video_analysis(video_output):
    df = pd.DataFrame(video_output,columns=['Prediction','Score'])
    df = df.groupby('Prediction').count()
    return df



# Load the cached assets
# model = load_model()
#class_names = load_class_names("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")


#VIDEO_URL = "https://www.youtube.com/watch?v=ZyMEXBKuGKY"
#st.video(VIDEO_URL)



#### BELOW IS APP ####

st.title("Unmasking Authentic Reactions in Dining Establishments")
#st.title("Test")

image_col, middle_col, output_col = st.columns([4,1,4])

with image_col:
    st.write('Upload a photo to analyse')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file != None:

        if uploaded_file.type == 'image/jpeg':
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(img_array, caption="Customer coming out of restaurant", use_column_width=True)

        elif uploaded_file.type == 'video/mp4':
            st.video(uploaded_file)
            g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
            temporary_location = "testout_simple.mp4"

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file
        
        else:
            st.write('Please upload only video or image for analysis. Other file types are not accepted.')

with middle_col:
    st.write('')

with output_col:
    st.write('Run predictions to classify image')
    st.header('')
    preds = st.slider("Select number of predictions to output",1,len(class_names),5)

    if st.button('Run predictions'):
        with st.spinner('Running Predictions'):
            time.sleep(5)
            if uploaded_file.type == 'image/jpeg':
                decoded_predictions = makes_preditions(uploaded_file,preds)

                st.write(f'The top {preds} predictions for classification are:')
                decode_and_print(decoded_predictions)
                st.write(':red[**Change the prediction number to show more/less options.]')

            elif uploaded_file.type == 'video/mp4':
                video_output = load_analyse_video(temporary_location)
                df_video = video_analysis(video_output)
                st.dataframe(df_video)


# In[ ]:




