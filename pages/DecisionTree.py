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

plt.plot(df['Close'])

df['Return'] = df['Adj Close'].pct_change(60).shift(-60)
list_of_features = ['High','Low','Close','Volume','Adj Close']
X= df[list_of_features]
y=np.where(df.Return > 0,1,0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=423)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

treeClassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6)
treeClassifier.fit(X_train, y_train)

y_pred = treeClassifier.predict(X_test)

from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred)
print(report)


from sklearn import tree
import graphviz
fig3 = plt.figure()
data = tree.export_graphviz(treeClassifier,filled=True,feature_names=list_of_features, class_names=np.array(['0','1']))
graphviz.Source(data)
st.pyplot(fig3)