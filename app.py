import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")
st.title("Employee Attrition Predictor")
st.write("Enter 8 employee details to estimate attrition probability.")

@st.cache_resource
def load_model():
    return joblib.load("attrition_model.sav")

model = load_model()

with st.form("prediction_form"):
    c1,c2=st.columns(2)
    with c1:
        age=st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        business_travel=st.selectbox("Business Travel", ["Travel_Rarely","Travel_Frequently","Non-Travel"])
        department=st.selectbox("Department", ["Research & Development","Sales","Human Resources"])
        job_role=st.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician",
            "Manufacturing Director","Healthcare Representative","Manager","Sales Representative",
            "Research Director","Human Resources"])
    with c2:
        marital_status=st.selectbox("Marital Status", ["Married","Single","Divorced"])
        monthly_income=st.number_input("Monthly Income", min_value=1000, max_value=200000, value=5000, step=500)
        overtime=st.selectbox("OverTime", ["No","Yes"])
        years_at_company=st.number_input("Years at Company", min_value=0, max_value=50, value=5, step=1)
    submitted=st.form_submit_button("Predict Attrition", type="primary")

if submitted:
    row=pd.DataFrame([{
        "Age":age,
        "BusinessTravel":business_travel,
        "Department":department,
        "JobRole":job_role,
        "MaritalStatus":marital_status,
        "MonthlyIncome":monthly_income,
        "OverTime":overtime,
        "YearsAtCompany":years_at_company
    }], columns=model.feature_names_in_)
    prediction=int(model.predict(row)[0])
    probability=float(model.predict_proba(row)[0][1])
    st.metric("Estimated attrition probability", f"{probability:.1%}")
    if prediction==1:
        st.warning("Model prediction: Attrition = Yes")
    else:
        st.success("Model prediction: Attrition = No")
    st.caption("This is a statistical estimate, not a definitive judgment about an employee.")
