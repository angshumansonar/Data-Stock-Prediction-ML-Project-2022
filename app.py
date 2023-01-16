import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from keras.models import load_model
import streamlit as st

st.title('Angshuman Sonar ML Project')
st.title('AI Stock Price Predictor : \nGOOG, TSLA, AAPL, JPM, WIPRO.NS, TCS.NS etc...')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = web.DataReader(user_input, data_source = 'yahoo', start = '2012-01-01', end = '2022-01-01')


#--------------------------------------------------------Describing Data
st.subheader('Data shown 2012 to 2022')
st.write(df.describe())

#--------------------------------------------------------Visualizations
st.subheader('Closing Price - Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price - Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price - Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

#--------------------------------------------------------Data Spliting
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

#--------------------------------------------------------Data Scaling Down
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#--------------------------------------------------------Load ML Model
model=load_model('my_keras_model.h5')


#--------------------------------------------------------Testing
past_100_days = data_training.tail(100)
final_df = past_100_days.append( data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


#--------------------------------------------------------Prediction
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


#--------------------------------------------------------Final Show
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='Original')
plt.plot(y_predicted,'r', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)