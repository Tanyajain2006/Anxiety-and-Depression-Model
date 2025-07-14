import gradio as gr
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import string

import pickle
import pandas as pd


# Function to clean text data: remove punctuation, lowercase
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text

# Load the trained model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Define the prediction function
def predict(text):

    # Clean the single text input using the same cleaning function
    cleaned_single_text = clean_text(text)

    # Transform the cleaned text using the loaded TF-IDF vectorizer
    single_text_tfidf = loaded_vectorizer.transform([cleaned_single_text])

    # Predict the label using the loaded model
    predicted_label = loaded_model.predict(single_text_tfidf)

    print(f"Original Text: {text}")
    print(f"Cleaned Text: {cleaned_single_text}")
    print(f"Predicted Label: {predicted_label[0]}")

    return "Anxiety/Depression" if predicted_label == 0 else "No Anxiety/Depression"

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter your text here..."),
    outputs="text",
    title="Anxiety and Depression Detector",
    description="Enter a social media comment or text input to check for signs of anxiety or depression.",
)

# Launch the app
interface.launch()
