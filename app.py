import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from main import read_data, preprocess_data, train_model, evaluate_model

st.set_page_config(page_title="No Code ML Model Training", layout="wide")
st.title("No Code ML Model Training")

uploaded_file = st.file_uploader("Select a dataset from the dropdown", type=["csv", "xlsx", "xls"])

model_trained = False
model_path = ""

if uploaded_file:
    df = read_data(uploaded_file)
    st.dataframe(df.head())

    with st.form("training_form"):
        st.markdown("### Model Configuration")
        target = st.selectbox("Select the Target Column", df.columns)
        scaler = st.selectbox("Select a scaler", ['standard', 'minmax'])
        model_type = st.selectbox("Select a Model", ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost'])
        model_name = st.text_input("Model name", value="my_model")
        submitted = st.form_submit_button("Train the Model")

        if submitted:
            try:
                X_train, X_test, y_train, y_test = preprocess_data(df, target, scaler)

                if model_type == "Logistic Regression":
                    model = LogisticRegression()
                elif model_type == "Random Forest":
                    model = RandomForestClassifier()
                elif model_type == "SVM":
                    model = SVC()
                elif model_type == "XGBoost":
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

                trained_model = train_model(X_train, y_train, model, model_name)
                accuracy = evaluate_model(trained_model, X_test, y_test)

                st.session_state['trained'] = True
                st.session_state['model_path'] = f"models/{model_name}.pkl"
                st.session_state['accuracy'] = accuracy

            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.get('trained'):
    st.success(f"Model trained and saved as `{st.session_state['model_path']}`")
    st.metric("Accuracy", f"{st.session_state['accuracy'] * 100:.2f}%")

    with open(st.session_state['model_path'], "rb") as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name=st.session_state['model_path'].split("/")[-1],
            mime="application/octet-stream"
        )


