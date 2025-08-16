import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize

tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
model = BertModel.from_pretrained("bert-large-cased")

emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

emotion_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}

def find_emotion(word):
    for key, values in emotion_mapping.items():
        if word in values:
            return key
    return "Not Found"


st.title("Go emotions ")
sentence = st.text_input("Enter a sentence: ")

if sentence:
    idx = 2 # remove this line after adding our model and uncomment below one 
    # idx = our_model_for_large_set_of_emotions.predict(embeddings)
    num =2 # assign real value after adding our model and uncomment below one 
    sentiment = "Positive" # replace with model value 
    st.write("#### specific emotion : "+ emotions[idx+1]) # assuming model returns value on 1 based indexing  
    st.write("#### generalized emotion : " + find_emotion(emotions[idx+1]))
    st.write("#### LRC_vc : ", num  )
    st.write("#### sentiment : ", sentiment )
