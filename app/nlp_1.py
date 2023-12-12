#before running the script, seek out the pip command in the notebook 
#that corresponds to the deployment script and run it in the terminal to get all the dependancies.

# script1.py
import streamlit as st

def run():
    st.title("DistilBERT fullset")
    import streamlit as st
import torch
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean and preprocess text
def clean_and_preprocess(text, use_stemming=True, use_lemmatization=True):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Load pre-trained model and tokenizer
model_path = "models/fine-tuned-toxic-ml"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Load data for EDA
# Assuming df_train is the DataFrame used for EDA
# You may need to adjust this based on your actual data
# df_train = pd.read_csv("your_data.csv")

# Distribution of Toxicity Labels
toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
label_counts = 6

# Streamlit App

st.title("Toxic Comment Classification")
st.write("DistilBERT model trained on the entire dataset")
comment = st.text_area("Type in a comment to determine its toxicity through a Natural Language Processing model")
# Display EDA graphs
#st.subheader("Distribution of Toxicity Labels")
#fig, ax = plt.subplots()
#label_counts.plot(kind='bar', ax=ax)
#ax.set_xlabel("Toxicity Labels")
#ax.set_ylabel("Count")
#st.pyplot(fig)

# User input
#comment = st.text_area("Enter your comment:")

# Make prediction on button click
# Make prediction on button click
if st.button("Determine toxicity"):
    if comment:
        # Apply preprocessing
        cleaned_comment = clean_and_preprocess(comment)
        
        # Tokenize and make prediction
        inputs = tokenizer(cleaned_comment, return_tensors='tf')
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = tf.nn.sigmoid(logits)
        predicted_labels = tf.cast(probabilities > 0.3, tf.int16)  # Adjust threshold as needed

        # Map predicted labels to toxicity types
        predicted_types = [toxicity_labels[i] for i, val in enumerate(predicted_labels.numpy().squeeze().tolist()) if val == 1]

        # Display results
        st.write(f"Processed comment: {cleaned_comment}")

        # Handle specific cases
        if not predicted_types:
            st.write("This is not a toxic comment.")
            st.write(f"Probabilities: {probabilities.numpy().squeeze().tolist()}")
        elif predicted_types == ['toxic']:
            st.write("This is a plain toxic comment.")
            st.write(f"Probabilities: {probabilities.numpy().squeeze().tolist()}")
        else:
            # Convert identity_hate labels for better readability
            predicted_types = [label.replace('_', ' ') if label == 'identity_hate' else label for label in predicted_types]
            
            st.write(f"Types of toxicity found within the comment: {', '.join(predicted_types)}")
            st.write(f"Probabilities: {probabilities.numpy().squeeze().tolist()}")
    else:
        st.warning("Please enter a comment.")

    # ... (rest of your script logic)
