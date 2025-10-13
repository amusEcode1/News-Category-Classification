import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier

import streamlit as st
import joblib

# Uses a pre-trained word embedding model (e.g., GloVe, Word2Vec).
class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.dim = model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = []
        for doc in X:
            # Filter out words not in embeddings
            vecs = [self.model[word] for word in doc if word in self.model]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(self.dim))
        return np.array(vectors)

# Define Keras Feedforward NN
def create_ffnn(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load saved pipeline
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

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
