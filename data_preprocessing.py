import joblib
import numpy as np
import pandas as pd

pca = joblib.load('model/pca.joblib')
scaler_age = joblib.load('model/scaler_Age_at_enrollment.joblib')
scaler_admission_grade = joblib.load('model/scaler_Admission_grade.joblib')
scaler_application_mode = joblib.load('model/scaler_Application_mode.joblib')
scaler_application_order = joblib.load('model/scaler_Application_order.joblib')
scaler_course = joblib.load('model/scaler_Course.joblib')
scaler_father_occupation = joblib.load('model/scaler_Fathers_occupation.joblib')
scaler_father_qualification = joblib.load('model/scaler_Fathers_qualification.joblib')
scaler_marital_status = joblib.load('model/scaler_Marital_status.joblib')
scaler_mother_occupation = joblib.load('model/scaler_Mothers_occupation.joblib')
scaler_mother_qualification = joblib.load('model/scaler_Mothers_qualification.joblib')
scaler_nationality = joblib.load('model/scaler_Nacionality.joblib')
scaler_previous_qualification_grade = joblib.load('model/scaler_Previous_qualification_grade.joblib')
scaler_previous_qualification_ = joblib.load('model/scaler_Previous_qualification.joblib')
scaler_curricular_unit_1st_sem_approved = joblib.load('model/scaler_Curricular_units_1st_sem_approved.joblib')
scaler_curricular_unit_1st_sem_credited = joblib.load('model/scaler_Curricular_units_1st_sem_credited.joblib')
scaler_curricular_unit_1st_sem_enrolled = joblib.load('model/scaler_Curricular_units_1st_sem_enrolled.joblib')
scaler_curricular_unit_1st_sem_evaluations = joblib.load('model/scaler_Curricular_units_1st_sem_evaluations.joblib')
scaler_curricular_unit_1st_sem_grade = joblib.load('model/scaler_Curricular_units_1st_sem_grade.joblib')
scaler_curricular_unit_2nd_sem_approved = joblib.load('model/scaler_Curricular_units_2nd_sem_approved.joblib')
scaler_curricular_unit_2nd_sem_credited = joblib.load('model/scaler_Curricular_units_2nd_sem_credited.joblib')
scaler_curricular_unit_2nd_sem_enrolled = joblib.load('model/scaler_Curricular_units_2nd_sem_enrolled.joblib')
scaler_curricular_unit_2nd_sem_evaluations = joblib.load('model/scaler_Curricular_units_2nd_sem_evaluations.joblib')
scaler_curricular_unit_2nd_sem_grade = joblib.load('model/scaler_Curricular_units_2nd_sem_grade.joblib')

pca_column = ['Curricular_units_1st_sem_credited',
              'Curricular_units_1st_sem_enrolled',
              'Curricular_units_1st_sem_evaluations',
              'Curricular_units_1st_sem_approved', 
              'Curricular_units_1st_sem_grade',
              'Curricular_units_2nd_sem_credited',
              'Curricular_units_2nd_sem_enrolled',
              'Curricular_units_2nd_sem_evaluations',
              'Curricular_units_2nd_sem_approved', 
              'Curricular_units_2nd_sem_grade']

def data_preprocessing(data):
    data = data.copy()
    df = pd.DataFrame()
    
    data['Application_mode'] = data['Application_mode'].map({'1st phase - general contingent':1,
                                                         'Ordinance No. 612/93':2,
                                                         '1st phase - special contingent (Azores Island)':5,
                                                         'Holders of other higher courses':7,
                                                         'Ordinance No. 854-B/99':10,
                                                         'International student (bachelor)':15,
                                                         '1st phase - special contingent (Madeira Island)':16,
                                                         '2nd phase - general contingent':17,
                                                         '3rd phase - general contingent':18,
                                                         'Ordinance No. 533-A/99, item b2) (Different Plan)':26,
                                                         'Ordinance No. 533-A/99, item b3 (Other Institution)':27,
                                                         'Over 23 years old':39,
                                                         'Transfer':42,
                                                         'Change of Course':43,
                                                         'Technological specialization diploma holders':44,
                                                         'Change of institution/course':51,
                                                         'Short cycle diploma holders':53,
                                                         'Change of institution/course (International)':57})



    data['Course'] = data['Course'].map({'Biofuel Production Technologies':33,
                                     'Animation and Multimedia Design':171,
                                     'Social Service (evening attendance)':8014,
                                     'Agronomy':9003,
                                     'Communication Design':9070,
                                     'Veterinary Nursing ':9085,
                                     'Informatics Engineering':9119,
                                     'Equinculture':9130,
                                     'Management':9147,
                                     'Social Service':9238,
                                     'Tourism':9254,
                                     'Nursing':9500,
                                     'Oral Hygiene':9556,
                                     'Advertising and Marketing Management':9670,
                                     'Journalism and Communication':9773,
                                     'Basic Education':9853,
                                     'Management (evening attendance)':9991})
    
    data['Previous_qualification'] = data['Previous_qualification'].map({"Secondary education":1, 
                                                                         "Higher education - bachelor's degree":2, 
                                                                         "Higher education - degree":3, 
                                                                         "Higher education - master's":4,
                                                                         "Higher education - doctorate":5, 
                                                                         "Frequency of higher education":6, 
                                                                         "12th year of schooling - not completed":9, 
                                                                         "11th year of schooling - not completed":10, 
                                                                         "Other - 11th year of schooling":12, 
                                                                         "10th year of schooling":14,
                                                                         "10th year of schooling - not completed":15, 
                                                                         "Basic education 3rd cycle (9th/10th/11th year) or equiv.":19, 
                                                                         "Basic education 2nd cycle (6th/7th/8th year) or equiv.":38, 
                                                                         "Technological specialization course":39,
                                                                         "Higher education - degree (1st cycle)":40, 
                                                                         "Professional higher technical course":42, 
                                                                         "Higher education - master (2nd cycle)":43})
    
    data['Marital_status'] = data['Marital_status'].map({'Single':1,
                                                       'Married':2,
                                                       'Widower':3,
                                                       'Divorced':4,
                                                       'Facto union':5,
                                                       'Legally separated':6})
    
    df['Marital_status'] = scaler_marital_status.transform(np.asarray(data['Marital_status']).reshape(-1,1))[0]
    df['Application_mode'] = scaler_application_mode.transform(np.asarray(data['Application_mode']).reshape(-1,1))[0]
    df['Application_order'] = scaler_application_order.transform(np.asarray(data['Application_order']).reshape(-1,1))[0]
    df['Course'] = scaler_course.transform(np.asarray(data['Course']).reshape(-1,1))[0]
    
    data['Daytime_evening_attendance'] = data['Daytime_evening_attendance'].map({'Daytime':1,
                                                                               'Evening':0})
    df['Daytime_evening_attendance'] = data['Daytime_evening_attendance']

    df['Previous_qualification_grade'] = scaler_previous_qualification_grade.transform(np.asarray(data['Previous_qualification_grade']).reshape(-1,1))[0]
    df['Previous_qualification'] = scaler_previous_qualification_grade.transform(np.asarray(data['Previous_qualification']).reshape(-1,1))[0]
    df['Admission_grade'] = scaler_admission_grade.transform(np.asarray(data['Admission_grade']).reshape(-1,1))[0]
    
    yes_no_column1 = ['Displaced', 'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date']
    
    for column in yes_no_column1:
        data[column] = data[column].map({'Yes':1,
                                       'No':0})
        
        df[column] = data[column]
        
    data['Gender'] = data['Gender'].map({'Male':1,
                                       'Female':0})
    df['Gender'] = data['Gender']
    
    data['Scholarship_holder'] = data['Scholarship_holder'].map({'Yes':1,
                                                                 'No':0})
        
    df['Scholarship_holder'] = data['Scholarship_holder']
        
    df['Age_at_enrollment'] = scaler_age.transform(np.asarray(data['Age_at_enrollment']).reshape(-1,1))[0]
    
    data['International'] = data['International'].map({'Yes':1,
                                                       'No':0})
        
    df['International'] = data['International']
    df['Curricular_units_1st_sem_without_evaluations'] = data['Curricular_units_1st_sem_without_evaluations']
    df['Curricular_units_2nd_sem_without_evaluations'] = data['Curricular_units_2nd_sem_without_evaluations']
    
    #df['Fathers_occupation'] = scaler_father_occupation.transform(np.asarray(data['Fathers_occupation']).reshape(-1,1))[0]
    #df['Fathers_qualification'] = scaler_father_qualification.transform(np.asarray(data['Fathers_qualification']).reshape(-1,1))[0]
    
    #df['Mothers_occupatin'] = scaler_mother_occupation.transform(np.asarray(data['Mothers_occupatin']).reshape(-1,1))[0]
    #df['Mothers_qualification'] = scaler_mother_qualification.transform(np.asarray(data['Mothers_qualification']).reshape(-1,1))[0]
    #df['Nacionality'] = scaler_nationality.transform(np.asarray(data['Nacionality']).reshape(-1,1))[0]

    
    data['Curricular_units_1st_sem_credited'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_1st_sem_credited']).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_enrolled'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_1st_sem_enrolled']).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_evaluations'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_1st_sem_evaluations']).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_approved'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_1st_sem_approved']).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_grade'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_1st_sem_grade']).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_credited'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_2nd_sem_credited']).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_enrolled'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_2nd_sem_enrolled']).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_evaluations'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_2nd_sem_evaluations']).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_approved'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_2nd_sem_approved']).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_grade'] = scaler_previous_qualification_grade.transform(np.asarray(data['Curricular_units_2nd_sem_grade']).reshape(-1,1))[0]
    
    df[['pc1', 'pc2']] = pca.transform(data[pca_column])
    
    return df
