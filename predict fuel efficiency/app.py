import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# ---- Sample dataset (you can replace with CSV) ----
# load the dataset
data=pd.read_csv("auto-mpg.csv")
data.head()
df = pd.DataFrame(data)
data['horsepower'].unique()
data['horsepower']=data['horsepower'].replace('?',pd.NA)
data.dropna(inplace=True)
data['horsepower']=data['horsepower'].astype(float)
print(data['horsepower'].astype(float))
# ---- Train model ----
# split the dataset
X=data.drop(['mpg','car name'],axis=1)
y=data['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain_scl=sc.fit_transform(X_train)
xtest_scl=sc.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(xtest_scl)
# evaluate the error
# o	Mean Absolute Error (MAE)  o Mean Squared Error (MSE)  o Root Mean Squared Error (RMSE)  o R² Score  
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared error  :",mse)

mae=mean_absolute_error(y_test,y_pred)
print("Mean absolute error  :",mae)

rmse=root_mean_squared_error(y_test,y_pred)
print("root Mean Squared error  :",rmse)

score=r2_score(y_test,y_pred)
print("r2 score   :",score)

# ---- Streamlit UI ----
st.title("Fuel efficency application")

cylinders   = st.number_input("•	Cylinders    :", min_value=1, max_value=10, step=1)
origin   = st.number_input("•	Origin    :", min_value=1, max_value=10, step=1)
displacement   = st.number_input("•	Displacement    :", min_value=1, max_value=10000, step=1)
hoursepower   = st.number_input("•	Horsepower    :", min_value=1, max_value=1000, step=1)
weight  = st.number_input("•	Weight   :", min_value=1, max_value=10000, step=1)
acceleration   = st.number_input("•	Acceleration    :", min_value=1, max_value=10000, step=1)
model_year   = st.number_input("•	Model Year    :", min_value=1, max_value=1000, step=1)

if st.button("Predict fuel efficency"):
    prediction = model.predict([[cylinders, origin, displacement, hoursepower,weight,acceleration, model_year  ]])[0]
    st.success(f"fuel efficency rate : ₹{prediction:.2f} rs")
    st.success(f"r2 score  : {score:.2f} ")
    st.success(f"RMSE  : {rmse:.2f} ")
    st.success(f"MSE  : {mse:.2f} ")
    st.success(f"MAE : {mae:.2f} ")
