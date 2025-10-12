import streamlit as st
import joblib

# Load saved pipeline
@st.cache_resource
def load_model():
    return joblib.load('news_model.joblib')

# Load the model
model = load_model()

# App title
st.title("📰 News Category Classification")
st.write("Classifies news articles into one of four categories: **World**, **Sports**, **Business**, or **Science/Tech**.")

# Text input
news_text = st.text_area("Enter a news article or headline:")

# Prediction button
if st.button("Classify"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([news_text])[0]
        st.success(f"Predicted Category: **{prediction}**")
