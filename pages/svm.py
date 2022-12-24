from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import yfinance as yf
plt.style.use('seaborn-darkgrid')

from sklearn.metrics import classification_report,confusion_matrix
import streamlit as st

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
  
# For data manipulation
import pandas as pd
import numpy as np
import seaborn as sns
  
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
%matplotlib inline
  
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="SVM")


st.markdown("# SVM")
st.sidebar.header("SVM")
st.write(
    """El contenido de la página permite visualizar resultados de predicción de precios de acciones utilizando el modelo SVM."""
)

ticker1 = st.text_input('Etiqueta de cotización', 'bvn')
st.write('La etiqueta de cotización actual es', ticker1)

bvn = yf.Ticker(ticker1)
hist = bvn.history(period="max", auto_adjust=True)
hist.head()
df = hist

df.info()



df['Stock Splits'].value_counts()



##ver la distribución porcentual de la columna Stock Splits
df['Stock Splits'].value_counts()/np.float(len(df))


df.isnull().sum()



round(df.describe(),2)


# draw boxplots to visualize outliers

plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df.boxplot(column='Open')
fig.set_title('')
fig.set_ylabel('Open')


plt.subplot(4, 2, 2)
fig = df.boxplot(column='High')
fig.set_title('')
fig.set_ylabel('High')


plt.subplot(4, 2, 3)
fig = df.boxplot(column='Low')
fig.set_title('')
fig.set_ylabel('Low')


plt.subplot(4, 2, 4)
fig = df.boxplot(column='Close')
fig.set_title('')
fig.set_ylabel('Close')


plt.subplot(4, 2, 5)
fig = df.boxplot(column='Volume')
fig.set_title('')
fig.set_ylabel('Volume')


plt.subplot(4, 2, 6)
fig = df.boxplot(column='Dividends')
fig.set_title('')
fig.set_ylabel('Dividends')


plt.subplot(4, 2, 7)
fig = df.boxplot(column='Stock Splits')
fig.set_title('')
fig.set_ylabel('Stock Splits')