import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Score Prediction", layout="centered")

st.title("💳 Credit Score Prediction App")
st.write("Predict customer credit score based on financial details")

# ===============================
# Load & Prepare Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("credit score.csv")

    # Cleaning
    df['Annual_Income'] = df['Annual_Income'].str.replace('_','').astype(float)
    df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(0)
    df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_','').astype(int)
    df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_','').astype(float)
    df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('_','').astype(float)
    df['Delay_from_due_date'] = pd.to_numeric(df['Delay_from_due_date'], errors='coerce').fillna(0)
    df['Num_of_Delayed_Payment'] = pd.to_numeric(
        df['Num_of_Delayed_Payment'].replace('_',''), errors='coerce'
    ).fillna(0)

    df['Credit_Mix'] = df['Credit_Mix'].replace('_','None')

    le = LabelEncoder()
    df['Credit_Mix'] = le.fit_transform(df['Credit_Mix'])
    df['Credit_Score'] = le.fit_transform(df['Credit_Score'])

    features = [
        'Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
        'Num_Credit_Card','Interest_Rate','Num_of_Loan',
        'Delay_from_due_date','Num_of_Delayed_Payment',
        'Credit_Mix','Outstanding_Debt','Monthly_Balance'
    ]

    X = df[features]
    y = df['Credit_Score']

    return X, y

X, y = load_data()

# ===============================
# Train Model
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

# ===============================
# User Input Section
# ===============================
st.subheader("🧾 Enter Customer Details")

annual_income = st.number_input("Annual Income", 0.0)
monthly_salary = st.number_input("Monthly Inhand Salary", 0.0)
bank_accounts = st.number_input("Number of Bank Accounts", 0)
credit_cards = st.number_input("Number of Credit Cards", 0)
interest_rate = st.number_input("Interest Rate", 0)
num_loans = st.number_input("Number of Loans", 0)
delay_days = st.number_input("Delay from Due Date (days)", 0)
delayed_payments = st.number_input("Number of Delayed Payments", 0)
credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
outstanding_debt = st.number_input("Outstanding Debt", 0.0)
monthly_balance = st.number_input("Monthly Balance", 0.0)

credit_mix_map = {"Bad":0, "Standard":1, "Good":2}

input_data = np.array([[
    annual_income,
    monthly_salary,
    bank_accounts,
    credit_cards,
    interest_rate,
    num_loans,
    delay_days,
    delayed_payments,
    credit_mix_map[credit_mix],
    outstanding_debt,
    monthly_balance
]])

input_scaled = scaler.transform(input_data)

# ===============================
# Prediction
# ===============================
if st.button("🔮 Predict Credit Score"):
    prediction = model.predict(input_scaled)[0]

    score_map = {0: "Poor", 1: "Standard", 2: "Good"}

    st.success(f"✅ Predicted Credit Score: **{score_map[prediction]}**")
