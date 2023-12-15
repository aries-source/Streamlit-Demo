import streamlit as st
import pandas as pd
import pickle 

def load_data():
    with  open("HR_model.pkl", 'rb') as file:
        data = pickle.load(file)

    return data

data= load_data()
model= data["model"]
le_salary = data['le_salary']

def show_predict_page():
    st.title("Exit Insight : A Machine Learning Predictor of Employee Retention")
    satisfaction = st.slider("Satisfaction level", 0.09,1.00,0.25)
    hours = st.slider("Average Monthly Hours", 96,310,100)
    Promotion = st.selectbox("Promotion within last 5 years",(0,1))
    salary = st.selectbox("Salary level",("low","medium","high"))
    btnAction = st.button("Predict")
      
    if btnAction:
       predictors = pd.DataFrame({
        "satisfaction_level":[satisfaction],
        "average_montly_hours" : [hours],
        "promotion_last_5years": [Promotion],
        "salary": [salary]
       })

       #st.dataframe(predictors)

       predictors.salary = le_salary.transform(predictors.salary)

       #st.dataframe(predictors)
       result= model.predict(predictors)
       
       st.write(f"The employee might {'leave.' if result==1 else 'stay.'}") 
       
       
     
       