import numpy as np
import joblib
import streamlit as st 
import re
import nltk
from nltk.corpus import stopwords   
from sklearn.feature_extraction.text import CountVectorizer     

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Function to clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens) 

# create UI
st.set_page_config(page_title="Sentiment Analysis App", page_icon=":speech_balloon:", layout="centered")
st.title("Sentiment Analysis App for movies")
st.markdown("Enter the movie review to predict sentiment:")
# Input field for user review
user_input = st.text_area("Review", height=200)
# Button to trigger prediction
if st.button("Predict"):
    if user_input:
        # Clean and preprocess the input
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction = model.predict(input_vector)
        
        # Display the result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Prediction is : {sentiment}")
    else:
        st.error("Please enter a review to analyze.")

