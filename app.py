import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="💳",
    layout="centered"
)

# ==================================================
# WEBSITE-STYLE GLOBAL CSS
# ==================================================
st.markdown("""
<style>
/* Page background */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    font-family: "Segoe UI", sans-serif;
}

/* Header section */
.header {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    color: white;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}
.header h1 {
    font-size: 40px;
    margin-bottom: 8px;
}
.header p {
    font-size: 18px;
    opacity: 0.95;
}

/* Main card */
.card {
    background: white;
    padding: 45px;
    border-radius: 26px;
    box-shadow: 0px 14px 45px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

/* Section titles */
.section-title {
    font-size: 30px;
    color: #1e293b;
    margin-bottom: 15px;
}

/* Inputs */
label {
    font-size: 18px !important;
    font-weight: 600;
    color: #334155;
}
input, select {
    font-size: 18px !important;
    border-radius: 14px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    font-size: 20px;
    font-weight: 700;
    padding: 14px 36px;
    border-radius: 18px;
    margin-top: 20px;
}

/* Result boxes */
.approved {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #14532d;
    padding: 28px;
    border-radius: 20px;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    margin-top: 25px;
}
.rejected {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #7f1d1d;
    padding: 28px;
    border-radius: 20px;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    margin-top: 25px;
}

/* Explanation */
.explain-box {
    background: #eef2ff;
    padding: 26px;
    border-radius: 20px;
    margin-top: 20px;
    border-left: 8px solid #6366f1;
    font-size: 18px;
    color: #1e293b;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 14px;
    color: #64748b;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="header">
    <h1>💳 Smart Loan Approval System</h1>
    <p>AI-powered loan eligibility prediction using Support Vector Machines</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA & TRAIN MODELS (LOGIC UNCHANGED)
# ==================================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

    cat_cols = [
        'Gender', 'Married', 'Dependents',
        'Education', 'Self_Employed',
        'Property_Area', 'Loan_Status'
    ]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    df.drop('Loan_ID', axis=1, inplace=True)

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Linear SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
        "Polynomial SVM": SVC(kernel='poly', degree=3, probability=True, class_weight='balanced'),
        "RBF SVM": SVC(kernel='rbf', C=0.8, gamma='scale',
                       probability=True, class_weight='balanced')
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, scaler, X.columns

models, scaler, feature_names = load_and_train()

# ==================================================
# MAIN FORM CARD
# ==================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Applicant Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Applicant Income", 0, step=500)
    credit = st.radio("Credit History", ["Yes", "No"])

with col2:
    loan_amt = st.number_input("Loan Amount", 0, step=500)
    employment = st.selectbox("Employment Status", ["Not Self Employed", "Self Employed"])

property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

credit = 1 if credit == "Yes" else 0
self_employed = 1 if employment == "Self Employed" else 0
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

st.markdown('<div class="section-title">🔧 Select SVM Kernel</div>', unsafe_allow_html=True)
kernel_choice = st.radio(
    "",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"],
    horizontal=True
)

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔍 Check Loan Eligibility"):

    input_data = np.zeros(len(feature_names))
    idx = {name: i for i, name in enumerate(feature_names)}

    input_data[idx['ApplicantIncome']] = income
    input_data[idx['LoanAmount']] = loan_amt
    input_data[idx['Credit_History']] = credit
    input_data[idx['Self_Employed']] = self_employed
    input_data[idx['Property_Area']] = property_area

    input_scaled = scaler.transform([input_data])

    model = models[kernel_choice]
    probs = model.predict_proba(input_scaled)[0]

    threshold = 0.6
    prediction = 1 if probs[1] >= threshold else 0
    confidence = probs[prediction]

    if prediction == 1:
        st.markdown('<div class="approved">✅ Loan Approved</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rejected">❌ Loan Rejected</div>', unsafe_allow_html=True)

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    st.markdown('<div class="explain-box">', unsafe_allow_html=True)
    st.subheader("🧠 Business Explanation")

    if prediction == 1:
        st.write(
            "Based on credit history, income stability, and employment pattern, "
            "the applicant is likely to repay the loan successfully."
        )
    else:
        st.write(
            "Based on credit risk indicators and repayment capacity, "
            "the applicant presents a higher risk of default."
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    © 2026 Smart Loan Approval System | Machine Learning & SVM
</div>
""", unsafe_allow_html=True)
