import streamlit as st
import pandas as pd
import os
import pickle
import sklearn 
import scipy
import numpy as np
import sklearn.externals as extjoblib
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#loading the trained model and components 
num_imputer = joblib.load('numerical_imputer.joblib')
cat_imputer = joblib.load('categorical_imputer.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')
dt_model = joblib.load('Final_model.joblib')

st.image("https://ultimahoraec.com/wp-content/uploads/2021/03/dt.common.streams.StreamServer-768x426.jpg")
st.title("Sales predictor app")

st.caption("This app predicts sales patterns of Cooperation Favorita over time in different stores in Ecuador.")

# Create the input fields
input_df = {}
col1,col2 = st.columns(2)
with col1:
    input_df["Year"] = st.number_input("Year",step=1)
    input_df["Month"] = st.slider("Month", min_value=1, max_value=12, step=1)
    input_df["Day"] = st.slider("Day", min_value=1, max_value=31, step=1)
    input_df["Quarter"] = st.slider("quater", min_value=1, max_value=4, step=1)
    input_df["week_of_year"] = st.slider("week_of_year", min_value=1, max_value=54, step=1)

with col2:   
    input_df["store_cluster"] = st.number_input("store_cluster", min_value=1,max_value=17,step=1)
    input_df["family"] = st.selectbox("family:", ["AUTOMOTIVE", "BEAUTY AND FASHION", "BEVERAGES AND LIQUOR", "FROZEN FOODS", "Grocery", "HOME AND KITCHEN", "HOME CARE AND GARDEN", "PET SUPPLIES", "SCHOOL AND OFFICE SUPPLIES"], index=2)
    input_df["events"] = st.selectbox("events:", ["Holiday", "No holiday"])
    input_df["oil_price"] =st.slider("Enter the current oil price",min_value=1.00,max_value=100.00,step=0.1)
    input_df["onpromotion"] = st.slider("Enter the no of item on promotion",min_value=1.00,max_value=30.0,step=0.1)
    cols = st.columns(2)
  # Create a button to make a prediction

if st.button("Predict"):
    # Convert the input data to a pandas DataFrame
        input_data = pd.DataFrame([input_df])


# Selecting categorical and numerical columns separately
        cat_columns = ["family", "events"]
        num_columns = input_data.drop(input_data[cat_columns], axis=1)


 # Encode the categorical columns
        encoder = OneHotEncoder(drop = "first", sparse=False)
        encoder.fit(input_data[cat_columns])
        encoded_features = encoder.transform(input_data[cat_columns])
        encoded_cat_col = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out().tolist())
        #new_df= pd.concat([new_df.reset_index(drop=True), encoded_cat_col], axis=1)

        #input_encoded_df = pd.DataFrame(encoder.transform(input_data[cat_columns]),
                                   #columns=encoder.get_feature_names(cat_columns))


#joining the cat encoded and num scaled
        final_df = pd.concat([encoded_cat_col, num_columns], axis=1)

# Make a prediction
        prediction =decision_tree.predict(final_df)[0]


# Display the prediction
        st.write(f"The predicted sales are: {prediction}.")
        input_df.to_csv("data.csv", index=False)
        st.table(input_df)


#Define columns
    cat = [col for col in new_df.columns if new_df[col].dtype == 'object']
    num = [col for col in new_df.columns if new_df[col].dtype != 'object']
    
   #Encoding on categorical columns.  
    encoder = OneHotEncoder(drop = "first", sparse=False)
    encoder.fit(df[cat])
    encoded_features = encoder.transform(df[cat])
    encoded_cat_col = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out().tolist())
    new_df= pd.concat([new_df.reset_index(drop=True), encoded_cat_col], axis=1)

    Prediction = decision_tree.predict(new_df)[0]
    st.subheader(f"The prediction is {Prediction}")
    input_df.to_csv("data.csv", index=False)
    st.table(input_df)

st.balloons()