
# 🧠 No-Code ML Model Training App

A **Streamlit-based** application that allows users to **train machine learning models without writing any code**. Upload your dataset, configure your model, and train it — all through a clean, interactive UI. You can also download the trained model for future use.

Streamlit App Link : https://no-code-auto-ml-nejcgnqzwjrakdjceq9q8s.streamlit.app/

---

## 🚀 Features

- 📁 **Dataset Upload**: Supports CSV, XLS, and XLSX files.
- 🎯 **Target Selection**: Choose your target column from the uploaded data.
- ⚙️ **Scaler Options**: Select between StandardScaler or MinMaxScaler.
- 🧠 **Model Options**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- 📊 **Data Preprocessing**: Automatic handling of missing values and categorical features.
- 📈 **Model Evaluation**: Displays accuracy after training.
- 💾 **Download Trained Model**: Save your model as a `.pkl` file for later use.

---

## 🖼️ Screenshots

### 🧩 Upload and Configure

![Screenshot 2025-06-22 131524](https://github.com/user-attachments/assets/c26143e2-2e42-42e2-afa7-e36e555982bf)

### 📈 Results and Download

![Screenshot 2025-06-22 131448](https://github.com/user-attachments/assets/120c2def-cede-4982-973a-ca365a772b5a)




---

## 📦 Installation

1. **Clone the repo**:

   ```bash
   git clone https://github.com/yourusername/no-code-ml-app.git
   cd no-code-ml-app
   ```

2. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

---

## 📚 Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- xgboost
- openpyxl (for Excel support)

---

## 📤 Upload Format

Ensure your dataset meets the following:
- One column must be the **target variable**.
- No target leakage (features shouldn't contain target information).
- Categorical and numerical features will be preprocessed automatically.

---

## 📥 Output

- Trained model saved as `models/your_model_name.pkl`.
- Accuracy metric displayed in the UI.
- Model available for direct download.

---

## 🛠️ Customization

You can extend this app by:
- Adding more model types (e.g., KNN, Decision Trees).
- Supporting hyperparameter tuning via UI.
- Adding evaluation metrics beyond accuracy (e.g., F1, ROC AUC).

---


## 👤 Author

Built with ❤️ by [Your Name](https://github.com/yourusername)

---
