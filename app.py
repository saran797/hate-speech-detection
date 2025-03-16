import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

rf_model = joblib.load("./models/random_forest_model.pkl")
svm_model = joblib.load("./models/svm_hate_speech_model.pkl")
xgb_model = joblib.load("./models/xgb_hsd_model.pkl")
lg_model = joblib.load("./models/logistic_regression_hsd_model.pkl")
nb_model = joblib.load("./models/naive_bayes_hsd_model.pkl")
st.title('Hate Speech Detection App')

count_vectorizer = joblib.load("./processor/count_vectorizer.pkl")
tfidf_transformer = joblib.load("./processor/tfidf_transformer.pkl")

accuracies = {
    "Random Forest": 98,  # Replace with your actual accuracy
    "SVM": 98,
    "XGBoost": 91,
    "Logistic Regression": 98,
    "Naive Bayes":95

}

models = {
    "Random Forest": rf_model,
    "SVM": svm_model,
    "Naive Bayes": nb_model,
    "XGBoost": xgb_model,
    "Logistic Regression":lg_model,
}
def pre_processing(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove special characters (keeping only letters, numbers, and spaces)
    text = re.sub(r'[^0-9A-Za-z \t]', ' ', text)
    # Remove extra spaces
    text = " ".join(text.split())
    return text

st.subheader("Model Accuracies",divider=True)
df_acc = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
st.table(df_acc)

selected_model= st.selectbox("Choose a model:", list(models.keys()),index=None,placeholder="Select a model",)
st.write("Model Selected : ",selected_model)

model=models[selected_model]
text=st.text_area("**Enter the text**")
if st.button("Classify"):
    processed_text = pre_processing(text)  # Preprocess input

    # Convert text to CountVectorizer format first
    vectorized_text = count_vectorizer.transform([processed_text])

    # Then apply TF-IDF transformation
    tfidf_text = tfidf_transformer.transform(vectorized_text)

    # Predict using the selected model
    prediction = model.predict(tfidf_text)[0]

    # Show result
    st.write(f"**Prediction:** {'Hate Speech' if prediction == 1 else 'Not Hate Speech'}")