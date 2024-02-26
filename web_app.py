import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

car=pd.read_csv("clean_data_car.csv")
brand=sorted(car['marque'].unique())

st.write('''
# Car Price Prediction using ML
Are you planning to sell your car ?
         
So lets try evaluating the price.
''')

st.sidebar.header("Input parameters")


def user_input():
    options_brand = sorted(car['marque'].unique())
    selected_option_brand = st.selectbox('Select a brand', options_brand )
    kms = st.number_input('Kilométrage', value=0, step=None)
    modeles_correspondants = car.loc[car['marque'] == selected_option_brand, 'info'].unique()
    selected_option_model = st.selectbox('Select a model', modeles_correspondants)
    options_type_fuel = sorted(car['type_carburant'].unique())
    selected_option_type_fuel = st.selectbox('Select a type of fuel', options_type_fuel )
    year = st.number_input('Choose a year', value=1975, step=None,min_value=1975,max_value=2024)

    data={
          'marque':selected_option_brand,
          'info':selected_option_model,
          'type_carburant':selected_option_type_fuel,
          'kilometrage':kms,
          'annee':year,
        
    }
    car_parameters=pd.DataFrame(data,index=[0])
    return car_parameters

df=user_input()
st.subheader("On veut prédire le prix de la voiture")
st.write(df)
X = car[['marque','info','type_carburant','kilometrage','annee']]
y = car['prix']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,test_size = 0.2,random_state=5432)
# Creating an object of Linear Regression
lm = LinearRegression()

# Fit the model using .fit() method
lm.fit(X_train, y_train)
y_test_pred = lm.predict(df)
st.subheader("Prix:")
st.write(y_test_pred)
