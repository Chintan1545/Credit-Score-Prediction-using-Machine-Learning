# 💳 Credit Score Prediction Web App (Streamlit)

## 📌 Project Overview
This project is an end-to-end **Credit Score Prediction system** built using Machine Learning and deployed as an **interactive Streamlit web application**.  
It predicts a customer's credit score category (Poor / Standard / Good) based on their financial and credit behavior data.

The goal is to automate credit risk assessment and support faster, data-driven financial decisions.

---

## 🎯 One-Line Explanation (Interview Ready)
> “I built and deployed a machine learning-based credit score prediction system using Streamlit that allows real-time credit risk assessment from user financial inputs.”

---

## 🧠 Problem Statement
Manual credit evaluation is time-consuming and error-prone. Financial institutions need an automated solution to evaluate customer creditworthiness using historical financial data.

This project solves that by:
- Cleaning and processing raw credit data
- Training machine learning models
- Deploying the best-performing model as a web app

---

## 🔄 How the System Works (Simple Flow)

1️⃣ **User Input**  
The user enters financial details such as income, loans, delayed payments, and outstanding debt.

2️⃣ **Data Preprocessing**  
Input data is scaled using `StandardScaler` to match the training distribution.

3️⃣ **Model Prediction**  
A trained **Random Forest Classifier** predicts the credit score category.

4️⃣ **Result Display**  
The predicted credit score (Poor / Standard / Good) is displayed instantly on the UI.

---

## 🏗 Architecture (One Line)
> “The pipeline preprocesses financial data, scales features, applies a Random Forest classifier, and serves predictions through a Streamlit interface.”

---

## 🛠 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Random Forest Classifier**
- **Streamlit**
- **Matplotlib & Seaborn (EDA)**

---

## 📊 Machine Learning Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier ✅ *(Best Performing)*
- Linear Discriminant Analysis (LDA)

---

## ✅ Why Random Forest?
- Handles non-linear relationships
- Robust to noisy data
- High accuracy with minimal tuning
- Works well with mixed financial features

---

## 📁 Project Structure
```bash
credit-score-streamlit/
│
├── app.py
├── credit score.csv
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Application

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
## Step 2: Run Streamlit App
```bash
streamlit run app.py
```

---

## 📦 requirements.txt
```bash
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 📈 Business Value

- This application helps financial institutions:
- Automate credit risk assessment
- Reduce manual evaluation time
- Improve loan approval accuracy
- Minimize default risk
- Scale credit evaluation for large user bases

---

## 🔮 Future Enhancements

- Model explainability using SHAP
- Hyperparameter tuning
- Database integration
- User authentication
- Cloud deployment (Streamlit Cloud / AWS)

## 👤 Author

Chintan Dabhi
