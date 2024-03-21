import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model weights from disk
with open('model.pkl', 'rb') as model_file:
    weights = pickle.load(model_file)

# Load the vectorizer from disk
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the prediction function
def predict(X, weight):
    return sigmoid(np.dot(X, weight))

# Streamlit app
def main():
    st.title("Department of Applied Mathematics and Statistics ")
    st.title("Email Spam Classifier")# Modified title
    st.text("By Team-3")  # Credit line

    # Create a text input for the user to enter an email
    email_text = st.text_area("Enter an email to classify:")

    # When the 'Predict' button is clicked, make a prediction on the input email
    if st.button('Predict'):
        # Transform the email into TF-IDF representation
        email_transformed = vectorizer.transform([email_text]).toarray()

        # Make a prediction on the transformed email
        prediction = predict(email_transformed, weights) >= 0.5

        # Display the prediction
        if prediction:
            st.write('This email is likely spam.')
        else:
            st.write('This email is likely not spam.')

if __name__ == '__main__':
    main()
