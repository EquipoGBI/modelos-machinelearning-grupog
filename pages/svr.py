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


st.set_page_config(page_title="SVR")

st.markdown("# SVR")
st.sidebar.header("SVR")
st.write(
    """En esta página podrás ver cómo funciona el modelo SVR en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'NFLX')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
hist

df = hist
df.info()

df = df.set_index(pd.DatetimeIndex(df['Date'].values))
df

future_days = 5

df[str(future_days)+'_Day_Stock_Forecast']=df[['Close']].shift(-future_days)
df[['Close',str(future_days)+'_Day_Stock_Forecast']]

X=np.array(df[['Close']])
X=X[:df.shape[0] - future_days]
print(X)

y=np.array(df[str(future_days)+'_Day_Stock_Forecast'])
y=y[:-future_days]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.00001)
svr_rbf.fit(x_train, y_train)

svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print('svr_rbf accuracy: ',svr_rbf_confidence)

svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)

print(y_test)

st.figure(figsize=(12,4))
st.plot(svm_prediction, label='Prediction', lw=2, alpha=.7)
st.plot(y_test, label='Actual', lw=2, alpha=.7)
st.title('Prediction vs Actual')
st.ylabel('Stock')
st.xlabel('Days')
st.legend()
st.xticks(rotation=45)
st.show()
