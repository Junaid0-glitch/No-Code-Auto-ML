
# 🧠 No-Code ML Model Training App

A **Streamlit-based** application that allows users to **train machine learning models without writing any code**. Upload your dataset, configure your model, and train it — all through a clean, interactive UI. You can also download the trained model for future use.

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

![Screenshot 1](screenshots/screenshot1.png)

### 📈 Results and Download

![Screenshot 2](screenshots/screenshot2.png)

> *Make sure to place your screenshots inside a `screenshots/` folder.*

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

## 📁 File Structure

```bash
no-code-ml-app/
├── app.py                   # Main Streamlit app
├── main.py                  # Utility functions: preprocess, train, evaluate
├── models/                  # Trained models saved as .pkl
├── screenshots/             # Screenshots for README
├── requirements.txt         # Python dependencies
└── README.md
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

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

Built with ❤️ by [Your Name](https://github.com/yourusername)

---
