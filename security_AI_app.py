
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from security_AI import TopicModel
from sklearn.pipeline import Pipeline
import mlflow
from models.model_builder import ExtSummarizer
from ext_sum import summarize
import torch
import nltk
import os
import urllib.request

st.write(""" 
# SECURITY AI APP

This app is used to show a **security event** occurence and **date of occurence**.

""")


st.subheader('Input News Url Here')
url = st.text_input("")

try:
    if url is not None:
        st.markdown(f"[*Read Original News*]({url})")
    else:
        st.write('Please input Urls Here')
except Exception:
    st.write('No Url Uploaded')

bow = pickle.load(open('BOW.sav','rb'))
model = mlflow.sklearn.load_model('LDA_model')
pipeline = Pipeline([('BOW', bow),('LDA', model)])
topic_model = TopicModel(pipeline)
text_corpus = topic_model._extractNewsContent(url)

if len(text_corpus) !=0:
    prediction = topic_model.predict(url,'Ner_Model2')

    st.subheader('Event Type')
    st.write(prediction.Topic.values)

    st.subheader('Event Location')
    st.write(prediction.Location.values)

    st.subheader('Event Date')
    st.write(prediction.Date.values)

else:
    st.write('There is no Url or the Url link is Invalid')


# st.write("News Text")
# st.write(text_corpus)

def download_model():
    nltk.download('popular')
    url = 'https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading checkpoint...")
        progress_bar = st.progress(0)
        with open('checkpoints/mobilebert_ext.pt', 'wb') as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
                        (counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

if not os.path.exists('checkpoints/mobilebert_ext.pt'):
    download_model()

# Summarize
@st.cache(suppress_st_warning=True)
def load_model(model_type):
    checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=model_type)
    return model

model = load_model('mobilebert')


input_fp = "raw_data/input.txt"
with open(input_fp, 'w') as file:
    file.write(text_corpus)

sum_level = st.radio("Output Length: ", ["Short", "Medium"])
max_length = 3 if sum_level == "Short" else 5
result_fp = 'results/summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=max_length)
st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)
