
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from security_AI import TopicModel
from sklearn.pipeline import Pipeline
import mlflow



st.write(""" 
# SECURITY AI APP

This app is used to show a **security event** occurence and **date of occurence**.

""")


st.sidebar.header('News Url')

uploaded_file = st.sidebar.file_uploader("Upload your input Security Event news Url", type = ['csv'])
try:
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.write('Please Upload a csv file of Urls')
except Exception:
    st.write('Upload a CSV file')


st.subheader('Event Url')

if uploaded_file is not None:
    st.write(input_df)


bow = pickle.load(open('BOW.sav','rb'))
model = mlflow.sklearn.load_model('LDA_model')
pipeline = Pipeline([('BOW', bow),('LDA', model)])

topic_model = TopicModel(pipeline)

try:
    prediction = topic_model.predict(input_df.Url.loc[0],'Ner_Model')

    st.subheader('Event Type')
    st.write(prediction.Topic.values)

    st.subheader('Event Location')
    st.write(prediction.Location.values)

    st.subheader('Event Date')
    st.write(prediction.Date.values)

    st.subheader('Event Prediction Table')
    st.write(prediction)

    st.subheader('Event News Text')
    text_corpus = topic_model._extractNewsContent(input_df.Url.iloc[0])
    st.write(text_corpus)
except Exception:
    st.write('No DataFrame Has been Uploaded')
