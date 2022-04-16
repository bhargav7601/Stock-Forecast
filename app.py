from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data #For scrapping data from Yahoo Finance website 
from keras.models import load_model
import streamlit as st


start = '2009-12-01'
end = '2019-12-31'
st.title('Stock Forecast')
user_input = st.text_input('Enter Stock Ticker','AAPL')

#data frame
df = data.DataReader(user_input, 'yahoo', start, end)
df = df.reset_index()
#df1 = data.DataReader(user_input, 'yahoo', start, end)

#Describing Data
st.subheader('Data from 2009 - 2019')
df = df.drop(['Date', 'Adj Close','Volume'], axis = 1)
# df = df.drop(['25%', '50%'], axis = 2)
st.write(df.describe())
#st.write(df1.describe())

#Visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'g')
# plt.plot(df.Open,'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 ma')
ma100 = df.Close.rolling(100).mean()
# fa100 = df.Open.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'b')
# plt.plot(fa100,'g')
plt.plot(df.Close,'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 ma and 200 ma ')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#Splitting Data into Training and Testing:

#70% of data for training and 
#30% of data for testing
data_training_close73 = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing_close73 = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

data_training_open73 = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])
data_testing_open73 = pd.DataFrame(df['Open'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler

scaler_close73 = MinMaxScaler(feature_range=(0,1))
data_training_array_close73 = scaler_close73.fit_transform(data_training_close73)

scaler_open73 = MinMaxScaler(feature_range=(0,1))
data_training_array_open73 = scaler_open73.fit_transform(data_training_open73)


#Loading Our Model 
model_close73 = load_model('keras_close73.h5')
model_open73 = load_model('keras_open73.h5')


#Testing Part
past_100_days = data_training_close73.tail(100)
past_100_days1 = data_training_open73.tail(100)
final_df = past_100_days.append(data_testing_close73, ignore_index=True)
final_df1 = past_100_days1.append(data_testing_open73, ignore_index=True)
input_data = scaler_close73.transform(final_df)
input_data1 = scaler_open73.transform(final_df1)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

a_test = []
b_test = []

for j in range(100, input_data1.shape[0]):
    a_test.append(input_data1[j-100: j])
    b_test.append(input_data1[j, 0])

a_test, b_test = np.array(a_test), np.array(b_test)


# making predictions 
y_predicted = model_close73.predict(x_test)
scaler_close73 = scaler_close73.scale_
scale_factor = 1/scaler_close73[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

b_predicted = model_open73.predict(a_test)
scaler_open73 = scaler_open73.scale_
scale_factor1 = 1/scaler_open73[0]
b_predicted = b_predicted * scale_factor1
b_test = b_test * scale_factor1

#Final Result Graph
st.subheader('Prediction vs Original 70:30')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price based on Closing Price')
# plt.plot(b_test, 'b', label = 'Original Price')
plt.plot(b_predicted, 'g', label = 'Predicted Price based on Open price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



#Splitting Data into Training and Testing:

#70% of data for training and 
#30% of data for testing
data_training_close82 = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing_close73 = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

data_training_open73 = pd.DataFrame(df['Open'][0:int(len(df)*0.80)])
data_testing_open73 = pd.DataFrame(df['Open'][int(len(df)*0.80):int(len(df))])

from sklearn.preprocessing import MinMaxScaler

scaler_close73 = MinMaxScaler(feature_range=(0,1))
data_training_array_close73 = scaler_close73.fit_transform(data_training_close82)

scaler_open73 = MinMaxScaler(feature_range=(0,1))
data_training_array_open73 = scaler_open73.fit_transform(data_training_open73)


#Loading Our Model 
model_close73 = load_model('keras_close82.h5')
model_open73 = load_model('keras_open82.h5')


#Testing Part
past_100_days = data_training_close82.tail(100)
past_100_days1 = data_training_open73.tail(100)
final_df = past_100_days.append(data_testing_close73, ignore_index=True)
final_df1 = past_100_days1.append(data_testing_open73, ignore_index=True)
input_data = scaler_close73.transform(final_df)
input_data1 = scaler_open73.transform(final_df1)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

a_test = []
b_test = []

for j in range(100, input_data1.shape[0]):
    a_test.append(input_data1[j-100: j])
    b_test.append(input_data1[j, 0])

a_test, b_test = np.array(a_test), np.array(b_test)


# making predictions 
y_predicted = model_close73.predict(x_test)
scaler_close73 = scaler_close73.scale_
scale_factor = 1/scaler_close73[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

b_predicted = model_open73.predict(a_test)
scaler_open73 = scaler_open73.scale_
scale_factor1 = 1/scaler_open73[0]
b_predicted = b_predicted * scale_factor1
b_test = b_test * scale_factor1

#Final Result Graph
st.subheader('Prediction vs Original 80:20')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price based on Closing Price')
# plt.plot(b_test, 'b', label = 'Original Price')
plt.plot(b_predicted, 'g', label = 'Predicted Price based on Open price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


