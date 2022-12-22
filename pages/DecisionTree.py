import os
import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import streamlit as st


start = datetime.datetime(2010,1,1)
end = datetime.datetime.now()

ticker = st.text_input('Etiqueta de cotización', 'GLD')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
hist

df = hist
df.info()

# Crea variables predictoras
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Guarda todas las variables predictoras en una variable X
X = df[['Open-Close', 'High-Low']]
X.head()

# Variables objetivas
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

treeClassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6)
treeClassifier.fit(X_train, y_train)

y_pred = treeClassifier.predict(X_test)

from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred)
st.write(report)



from sklearn import tree
import graphviz
fig3 = plt.figure()
data = tree.export_graphviz(treeClassifier,filled=True,feature_names=list_of_features, class_names=np.array(['0','1']))
graphviz.Source(data)
st.pyplot(fig3)