from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import yfinance as yf
plt.style.use('seaborn-darkgrid')
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report,confusion_matrix
import streamlit as st


st.set_page_config(page_title="KNN")

st.markdown("# KNN")
st.sidebar.header("KNN")
st.write(
    """El contenido de la página permite visualizar resultados de predicción de precios de acciones utilizando el modelo KNN."""
)

ticker = st.text_input('Etiqueta de cotización', 'INTC')
st.write('La etiqueta de cotización actual es', ticker)

intc = yf.Ticker(ticker)
hist = intc.history(period="max", auto_adjust=True)
hist.head()

df = hist

df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

X = df[['Open-Close', 'High-Low']]
X.head()

y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
y

split_percentage = 0.7
split = int(split_percentage*len(df))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

print("Predicciones del clasificador:")
test_data_predicted = knn.predict(X_test)
print(test_data_predicted)
st.write(test_data_predicted)
print("Resultados esperados:")
print(y_test)
st.write(y_test)

# Datos predecidos 
st.write("Dataframe con los resultados predecidos")
df['Predicted_Signal'] = knn.predict(X)

print(accuracy_score(test_data_predicted, y_test))
# Precisión del modelo
st.write(accuracy_score(test_data_predicted, y_test))

tasa_error = []
for i in range(1,40):
  knn_g = KNeighborsClassifier(n_neighbors=i)
  knn_g.fit(X_train,y_train)
  pred_i = knn_g.predict(X_test)
  tasa_error.append(np.mean(pred_i != y_test))

fig = plt.figure(figsize=(10,6),dpi=250)
plt.plot(range(1,40),tasa_error,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Tasa de Error vs. Valor de K')
plt.xlabel('K')
plt.ylabel('Tasa de Error')
st.pyplot(fig)

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('CON K=19')
print(classification_report(y_test,pred))