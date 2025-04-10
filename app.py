import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.neural_net_model import build_model, train_model
from utils.evaluator import evaluate_model
from utils.visualizer import plot_training_history

st.title("UCLA Admission Neural Network Classifier")
st.markdown("Predict admission chances using a simple 2-layer neural network.")

# Upload or use default
uploaded_file = st.file_uploader("Upload UCLA dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully.")
else:
    df = load_data("data/Admission.csv")
    st.info("Using default dataset from `data/` folder.")

st.write("### Sample Data")
st.dataframe(df.head())

if st.button("Train Neural Network"):
    with st.spinner("Training the model..."):
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = build_model(input_shape=X_train.shape[1])
        model, history = train_model(model, X_train, y_train, epochs=50, verbose=0)

        results = evaluate_model(model, X_test, y_test)
        st.success("Model training completed.")
        st.write(f"**Accuracy**: `{results['accuracy']:.2f}`")
        st.write("### Classification Report")
        st.text(results["classification_report"])
        st.write("### Confusion Matrix")
        st.write(results["confusion_matrix"])

        st.write("### Training History")
        fig = plot_training_history(history)
        st.pyplot(fig)
