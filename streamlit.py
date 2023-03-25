import streamlit as st
import pandas as pd
import os
import pickle
#import sklearn 
#import scipy
import numpy as np
#import joblib

# loading the trained model 
try:
   with open("Ts_model.sav", "rb") as f:
        model = pickle.load(f)
except IndexError as e:
    print(e)

# first line after the importation section
st.set_page_config(page_title="Sales predictor app", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))

@st.cache_resource()  # stop the hot-reload to the function just bellow

def setup(df):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            date=[],
            Year=[],
            Month=[],
            day=[],
            week_of_year=[],
            onpromotion=[],
            store_cluster=[],
            family=[],
            events=[],
            oil_price=[],    
        )
    ).to_csv(df, index=False)
    
df = os.path.join(DIRPATH, "df.csv")
setup(df)

#predict = model.prediction(data)
#return predict

ml_components_dict = setup(df)
def predict(df):
    cat_cols = ml_components_dict['cat']
    encoder = ml_components_dict['OneHotEncoder']
    prediction = model.predict(df)
    return prediction
    

# prediction execution


#labels = ml_components_dict['labels']
#num_cols = ml_components_dict['num']
#num_imputer = ml_components_dict['num_imputer']
#cat_imputer = ml_components_dict['cat_imputer']
#scaler = ml_components_dict['StandardScaler']
#model = ml_components_dict['model']


#creating the interface

st.image("https://ultimahoraec.com/wp-content/uploads/2021/03/dt.common.streams.StreamServer-768x426.jpg")
st.title("Sales predictor app")

st.caption("This app predicts sales patterns of Cooperation Favorita over time in different stores in Ecuador.")

form = st.form(key="information", clear_on_submit=True)

with form:

    cols = st.columns((1, 1))
    date = cols[0].date_input("date")
    onpromotion = cols[0].selectbox("onpromotion:", ["Yes", "No"])
    store_cluster = cols[1].number_input("store_cluster", min_value=1,max_value=17,step=1)
    family = cols[0].selectbox("family:", ["AUTOMOTIVE", "BEAUTY AND FASHION", "BEVERAGES AND LIQUOR", "FROZEN FOODS", "Grocery", "HOME AND KITCHEN", "HOME CARE AND GARDEN", "PET SUPPLIES", "SCHOOL AND OFFICE SUPPLIES"], index=2)
    events = cols[1].selectbox("events:", ["Holiday", "No holiday"])
    oil_price=st.slider("Enter the current oil price",min_value=1.00,max_value=100.00,step=0.1)
    
    cols = st.columns(2)

    submitted = st.form_submit_button(label="Predict")
    

if submitted:
    st.success("Thanks!")
    pd.read_csv(df).append(
        dict(
            date=date,
            Year=date.year,
            Month=date.month,
            day=date.day,
            week_of_year=date.isocalendar().week,
            onpromotion=onpromotion,
            store_cluster=store_cluster,
            family=family,
            events=events,
            sales=predict
        ),
        ignore_index=True,
    ).to_csv(df, index=False)
    st.balloons()

expander = st.expander("See all records")
with expander:
    df_new = pd.read_csv(df)
    st.dataframe(df_new)