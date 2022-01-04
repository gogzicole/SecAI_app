
import streamlit as st
import numpy as np
import pickle
from security_AI import TopicModel
from sklearn.pipeline import Pipeline
import mlflow
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
    news,topic,location,date = topic_model.predict(url,'Ner_Model2')

    st.subheader('Event Type')
    st.write(topic)

    st.subheader('Event Location')
    st.write(location)

    st.subheader('Event Date')
    st.write(date)

else:
    st.write('There is no Url or the Url link is Invalid')


st.write("News Text")
st.write(text_corpus)
