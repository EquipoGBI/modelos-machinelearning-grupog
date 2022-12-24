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
df



##ver la distribución porcentual de la columna Stock Splits
df['Stock Splits'].value_counts()/np.float(len(df))
df

df.isnull().sum()
df


round(df.describe(),2)


#Declarar vector de características y variable de destino
X = df.drop(['Stock Splits'], axis=1)
y = df['Stock Splits']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


X_train.shape, X_test.shape

cols = X_train.columns


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# import SVC classifier
from sklearn.svm import SVC


# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
svc.fit(X_train,y_train)
svc

# make predictions on test set
y_pred=svc.predict(X_test)

# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0) 


# fit classifier to training set
svc.fit(X_train,y_train)
svc

# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0) 


# fit classifier to training set
svc.fit(X_train,y_train)
svc

# make predictions on test set
y_pred=svc.predict(X_test)
y_pred


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0) 


# fit classifier to training set
linear_svc.fit(X_train,y_train)


# make predictions on test set
y_pred_test=linear_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0) 


# fit classifier to training set
linear_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0) 


# fit classifier to training set
linear_svc1000.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc1000.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



y_pred_train = linear_svc.predict(X_train)
y_pred_train


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))

st.write("Plot Strategy Returns vs Original Returns")
fig = plt.figure()
plt.plot(y_test, color='red')
plt.plot(y_pred_test, color='blue')
st.pyplot(fig)

report = classification_report(y_test, y_pred_train)
print(report)