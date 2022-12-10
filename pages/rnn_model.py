
import datetime as dt
import yfinance as yf
import warnings
import streamlit as st
from math import sqrt
# Machine learning

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# To ignore warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="PEN")

st.markdown("# PEN")
st.sidebar.header("PEN")
st.write(
    """En esta página podrás ver cómo funciona el modelo PEN en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'PEN')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
hist = tic.history(period="max", auto_adjust=True)

hist.to_csv('bse.csv', index=False)
hist

hist = tic.history(period="max", auto_adjust=True)
hist

df = hist
df.info()

# data de comparacion test
testdf = yf.download("PEN", start="2022-03-31",
                     end=dt.datetime.now(), progress=False)
testdf
# Realizar la preparacion de dato de RNN model
training_set = hist.iloc[:, 1:2].values
training_set
# Transformaciones de mminimo y maximo
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []

for i in range(60, 566):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train,  y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# inicializar modelo rnn
regressor = Sequential()

# añadir el primer LSTM Layer y regularizaciones
regressor.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# añadir un segundo lstm y regularizar el dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# añadir un tercer lstm y regularizar el dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# añadir un cuarto lstm y regularizar el dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
# Entrenando la data en RNN set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
dataset_total = pd.concat((hist['Open'], testdf['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(testdf) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 213):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = testdf.iloc[:, 1:2].values
# Visualizar los resultados obtenidos
plt.plot(real_stock_price, color='red', label='Real stock de Soles')
plt.plot(predicted_stock_price, color='blue',
         label='Prediccion del stock de Soles')
st.write("Prediccion del precio de sol")
plt.title('Prediccion del precio de sol')
plt.xlabel('Time')
plt.ylabel('Precio stock')
fig = plt.figure()
plt.legend()
st.pyplot(fig)

