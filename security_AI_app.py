
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from security_AI import TopicModel
from sklearn.pipeline import Pipeline
import mlflow
from models.model_builder import ExtSummarizer
from ext_sum import summarize
#import torch


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


st.write("News Text")
st.write(text_corpus)

# Summarize
# @st.cache(suppress_st_warning=True)
# def load_model(model_type):
#     checkpoint = torch.load(f'checkpoints/{model_type}.pt', map_location='cpu')
#     model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=model_type)
#     return model

# model = load_model('mobilebert')

# sum_level = st.radio("Output Length: ", ["Short", "Medium"])
# max_length = 3 if sum_level == "Short" else 5
# result_fp = 'results/summary.txt'
# summary = summarize(text_corpus, result_fp, model, max_length=max_length)
# st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
# st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)
        