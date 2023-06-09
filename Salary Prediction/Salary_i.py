import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st
import pandas as pd

"""
Loading the Model
"""

model_i= load_model(r'C:\Users\shreyansh verma\Documents\Work\VS Code Projects\Machine Learning\Practice\Salary Prediction\Model_Save\Salary_Model_Save.h5')

"""
Input Data
"""
filepath= r'C:\Users\shreyansh verma\Documents\Work\VS Code Projects\Machine Learning\Practice\Salary Prediction\Salary_Data.csv'
salary_data= pd.read_csv(filepath)
job_list=(salary_data['Job Title'].unique())


age= st.number_input("Age")
xp= st.number_input("Year of Expirience")
gender= st.selectbox("Gender", ('Male', 'Female','Other'))
edu= st.selectbox("Highest Education", ("High School", "master", "bachelor", "phd"))
job= st.selectbox("Job", (job_list))

"""
UI DF
"""

UI_df_path= r'C:\Users\shreyansh verma\Documents\Work\VS Code Projects\Machine Learning\Practice\Salary Prediction\UI_df.csv'
UI_df= pd.read_csv(UI_df_path)
UI_df.all= 0


if age == 0:
    age=0.1
if xp == 0:
    xp= 0.1
UI_df['age and work']= age/xp


gender= 'Gender_'+gender
UI_df[gender]= 1.0


UI_df[edu]= 1.0


UI_df[job]= 1.0


done= st.button('Done')



if done == True:
    UI_df= UI_df.astype(float)

    for col in UI_df:
        UI_df = UI_df.drop(UI_df[UI_df[col] == float('inf')].index)
        
    model_i.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    tf.config.run_functions_eagerly(True)
    pred = model_i.evaluate(UI_df)
    st.title(pred)
