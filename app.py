import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing
from prediction import prediction

data = pd.DataFrame([[0]])
data.columns = ['a']

col1, col2 = st.columns(2)

with col1:
    
    Marital = st.selectbox(label='Marital Status', options=['Single',
                                                       'Married',
                                                       'Widower',
                                                       'Divorced',
                                                       'Facto union',
                                                       'Legally separated'])
    data["Marital_status"] = Marital
    
with col2:
    
    Attendance = st.selectbox(label='Attendance', options=['Daytime', 'Evening'])
    data["Daytime_evening_attendance"] = Attendance
 
col1, col2, col3 = st.columns(3)

with col1:
    
    Application_mode = st.selectbox(label='Application Mode', options=["1st phase - general contingent", 
                                                                       "Ordinance No. 612/93", 
                                                                       "1st phase - special contingent (Azores Island)", 
                                                                       "Holders of other higher courses", "Ordinance No. 854-B/99",
                                                                       "International student (bachelor)", 
                                                                       "1st phase - special contingent (Madeira Island)", 
                                                                       "2nd phase - general contingent", 
                                                                       "3rd phase - general contingent", 
                                                                       "Ordinance No. 533-A/99, item b2 (Different Plan)", 
                                                                       "Ordinance No. 533-A/99, item b3 (Other Institution)"
                                                                       "Over 23 years old", "Transfer", "Change of course", 
                                                                       "Technological specialization diploma holders",
                                                                       "Change of institution/course", 
                                                                       "Short cycle diploma holders", 
                                                                       "Change of institution/course (International)"])
    data["Application_mode"] = Application_mode
 
with col2:
    
    Application_order = int(st.number_input(label='Application Order', value=4))
    data["Application_order"] = Application_order
 
with col3:
    
    Course = st.selectbox(label='Course', options=["Biofuel Production Technologies", "Animation and Multimedia Design", "Social Service (evening attendance)", "Agronomy",
                                                                       "Communication Design", "Veterinary Nursing ", "Informatics Engineering", 
                                                                       "Equinculture ", "Management ", "Social Service"
                                                                       "Tourism ", "Nursing ", "Oral Hygiene", "Advertising and Marketing Management",
                                                                       "Journalism and Communication", "Basic Education", "Management (evening attendance)"])
    data["Course"] = Course
 
 
col1, col2, col3 = st.columns(3)
 
with col1:
    
    Previous_qualification = st.selectbox(label='Previous Qualification', options=["Secondary education", "Higher education - bachelor's degree", "Higher education - degree", "Higher education - master's",
                                                                                   "Higher education - doctorate", "Frequency of higher education", "12th year of schooling - not completed", 
                                                                                   "11th year of schooling - not completed ", "Other - 11th year of schooling", "10th year of schooling"
                                                                                   "10th year of schooling - not completed ", "Basic education 3rd cycle (9th/10th/11th year) or equiv.", "Basic education 2nd cycle (6th/7th/8th year) or equiv.", "Technological specialization course",
                                                                                   "Higher education - degree (1st cycle)", "Professional higher technical course", "Higher education - master (2nd cycle)"])
    data["Previous_qualification"] = Previous_qualification
 
with col2:
    
    Previous_qualification_grade = int(st.number_input(label='Previous qualification grade', value=100))
    data["Previous_qualification_grade"] = Previous_qualification_grade
 
with col3:
    
    Admission_grade = int(st.number_input(label='Admission grade', value=7))
    data["Admission_grade"] = Admission_grade
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:
    
    Displaced = st.selectbox(label='Displaced', options=['Yes', 'No'])
    data["Displaced"] = Displaced
 
with col2:
    
    Need = st.selectbox(label='Educational_special_needs', options=['Yes', 'No'])
    data["Educational_special_needs"] = Need
 
with col3:
    
    Debtor = st.selectbox(label='Debtor', options=['Yes', 'No'])
    data["Debtor"] = Debtor
 
with col4:
    
    Tuition = st.selectbox(label='Tuition Fees Up to Date', options=['Yes', 'No'])
    data["Tuition_fees_up_to_date"] = Tuition
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:
    
    Gender = st.selectbox(label='Gender', options=['Male', 'Female'])
    data["Gender"] = Gender
 
with col2:
    
    Scholarship = st.selectbox(label='Scholarship Holder', options=['Yes', 'No'])
    data["Scholarship_holder"] = Scholarship
 
with col3:
    
    Ages = float(st.number_input(label='Ages at Enrollment', value=20))
    data["Age_at_enrollment"] = Ages
    
with col4:
    
    International = st.selectbox(label='International', options=['Yes', 'No'])
    data["International"] = International
    
col1, col2, col3 = st.columns(3)
 
with col1:
    
    Curricular_units_1st_sem_credited = float(st.number_input(label='Curricular_units_1st_sem_credited', value=0))
    data["Curricular_units_1st_sem_credited"] = Curricular_units_1st_sem_credited
 
with col2:
    
    Curricular_units_1st_sem_enrolled = float(st.number_input(label='Curricular_units_1st_sem_enrolled', value=0))
    data["Curricular_units_1st_sem_enrolled"] = Curricular_units_1st_sem_enrolled
 
with col3:
    
    Curricular_units_1st_sem_evaluations = float(st.number_input(label='Curricular_units_1st_sem_evaluations', value=0))
    data["Curricular_units_1st_sem_evaluations"] = Curricular_units_1st_sem_evaluations
    
col1, col2, col3 = st.columns(3)
    
with col1:
    
    Curricular_units_1st_sem_approved = float(st.number_input(label='Curricular_units_1st_sem_approved', value=0))
    data["Curricular_units_1st_sem_approved"] = Curricular_units_1st_sem_approved
    
with col2:
    
    Curricular_units_1st_sem_grade = float(st.number_input(label='Curricular_units_1st_sem_grade', value=0))
    data["Curricular_units_1st_sem_grade"] = Curricular_units_1st_sem_grade
    
with col3:
    
    Curricular_units_1st_sem_without_evaluations = float(st.number_input(label='Curricular_units_1st_sem_without_evaluations', value=0))
    data["Curricular_units_1st_sem_without_evaluations"] = Curricular_units_1st_sem_without_evaluations

    
col1, col2, col3 = st.columns(3)
 
with col1:
    
    Curricular_units_2nd_sem_credited = float(st.number_input(label='Curricular_units_2nd_sem_credited', value=0))
    data["Curricular_units_2nd_sem_credited"] = Curricular_units_2nd_sem_credited
 
 
with col2:
    
    Curricular_units_2nd_sem_enrolled = float(st.number_input(label='Curricular_units_2nd_sem_enrolled', value=0))
    data["Curricular_units_2nd_sem_enrolled"] = Curricular_units_2nd_sem_enrolled
 
with col3:
    
    Curricular_units_2nd_sem_evaluations = float(st.number_input(label='Curricular_units_2nd_sem_evaluations', value=0))
    data["Curricular_units_2nd_sem_evaluations"] = Curricular_units_2nd_sem_evaluations
    
col1, col2, col3 = st.columns(3)
    
with col1:
    
    Curricular_units_2nd_sem_approved = float(st.number_input(label='Curricular_units_2nd_sem_approved', value=0))
    data["Curricular_units_2nd_sem_approved"] = Curricular_units_2nd_sem_approved
    
with col2:
    
    Curricular_units_2nd_sem_grade = float(st.number_input(label='Curricular_units_2nd_sem_grade', value=0))
    data["Curricular_units_2nd_sem_grade"] = Curricular_units_2nd_sem_grade
    
with col3:
    
    Curricular_units_2nd_sem__without_evaluations = float(st.number_input(label='Curricular_units_2nd_sem_without_evaluations', value=0))
    data["Curricular_units_2nd_sem_without_evaluations"] = Curricular_units_2nd_sem__without_evaluations

data.drop('a', axis=1)

#View Raw Data
with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=20)
    
#Prediciton Button
if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    print(new_data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Result: {}".format(prediction(new_data)))