import streamlit as st
import pickle
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
stopwords=pickle.load(open('stopwords.pkl','rb'))

st.title("Email/SMS Classifier")

input_sms=st.text_area("Enter the message")

import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = text.split(' ')

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

if st.button("Predict"):
    # 1.preprocess
    transform_sms=transform_text(input_sms)
    #2.vectorise
    vector_input=tfidf.transform([transform_sms])
    #3.predict
    result=model.predict(vector_input)[0]
    #4.display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")