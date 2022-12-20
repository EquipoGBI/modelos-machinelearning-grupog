import yfinance as yf
import warnings
import streamlit as st
import os
import datetime

# Machine learning
from sklearn.svm import SVR
from sklearn.metrics import accuracy_scor, classification_report
from sklearn.tree import DecisionTreeClassifier

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# To ignore warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Decision Tree")

st.markdown("# Decision Tree")
st.sidebar.header("Decision Tree")
st.write(
    """En esta página podrás ver cómo funciona el modelo Decision Tree en la predicción del mercado de valores"""
)

start = datetime.datetime(2010,1,1)
end = datetime.datetime.now()

df = pd.read_csv('GLD.csv')
df

plt.plot(df['Close'])

df['Return'] = df['Adj Close'].pct_change(60).shift(-60)
list_of_features = ['High','Low','Close','Volume','Adj Close']
X= df[list_of_features]
y=np.where(df.Return > 0,1,0)

split_percentage = 0.7
split = int(split_percentage*len(df))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


treeClassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6)
treeClassifier.fit(X_train, y_train)
y_pred = treeClassifier.predict(X_test)

report = classification_report(y_test,y_pred)
st.write(report)