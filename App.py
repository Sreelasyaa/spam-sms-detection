
import streamlit as st
import joblib
import pandas as pd

# Load models and vectorizer
nb_model = joblib.load('nb_model.pkl')
lr_model = joblib.load('lr_model.pkl')
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define function to make predictions
def predict(model, message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit app
st.title("Spam SMS Detection")

# User input
message = st.text_area("Enter a message to classify:")

# Model selection
model_choice = st.selectbox("Choose a model for prediction:", ["Naive Bayes", "Logistic Regression", "Support Vector Machine"])

if st.button("Predict"):
    if model_choice == "Naive Bayes":
        prediction = predict(nb_model, message)
    elif model_choice == "Logistic Regression":
        prediction = predict(lr_model, message)
    elif model_choice == "Support Vector Machine":
        prediction = predict(svm_model, message)
    
    st.write(f"Prediction: **{prediction}**")

# Display model performance
st.header("Model Performance")

model_performance = {
    "Model": ["Naive Bayes", "Logistic Regression", "Support Vector Machine"],
    "Accuracy": [0.9884, 0.9872, 0.9797],
    "Precision": [0.9787, 0.9756, 0.9452],
    "Recall": [0.9839, 0.9845, 0.9826],
    "F1 Score": [0.9813, 0.9800, 0.9635]
}

performance_df = pd.DataFrame(model_performance)
st.dataframe(performance_df)
