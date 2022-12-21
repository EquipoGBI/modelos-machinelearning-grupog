import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# To ignore warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="LSTM")

st.markdown("# SVR")
st.sidebar.header("SVR")
st.write(
    """En esta página podrás ver cómo funciona el modelo SVR en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'AAPL')
st.write('La etiqueta de cotización actual es', ticker)
st.write('Apple Inc. (AAPL)') 
tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
hist
###########################
import time
import math
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
###########################
st.write("Dataframe obtenidode AAPL")
df = hist
df.head()

import pandas as pd
import matplotlib.pyplot as plt 
import quandl 
from sklearn.linear_model import LinearRegression

df.isnull().sum()

st.write('Mapa de calor') 
import seaborn as sns
plt.figure(1 , figsize = (17 , 8))
cor = sns.heatmap(df.corr(), annot = True)

x = df.loc[:,'high':'close']
y = df.loc[:,'open']

y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 101)

LR = LinearRegression()

LR.fit(x_train,y_train)

LR.score(x_test,y_test)

#Evaluación del Modelo
#Se evalua el modelo comprobando sus coeficientes.
#Se imprime el interceptor
#print(LR.intercept_)

coeff_df = pd.DataFrame(LR.coef_,x.columns,columns=['Coeficiente'])
coeff_df

x_test.head()

predictions = LR.predict(x_test)
predictions

plt.scatter(y_test,predictions)

sns.displot((y_test-predictions),bins=50);

#MÉTRICAS

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(predictions,color='red')