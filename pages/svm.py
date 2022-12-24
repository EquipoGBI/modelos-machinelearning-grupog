import yfinance as yf
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Get stock data for BVN from yahoo finance
bvn = yf.Ticker("BVN").history(period="max")
bvn 

# Select relevant columns
X = bvn[['Open', 'Close', 'Volume']]
y = bvn['Close'].shift(-1) > bvn['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a StandardScaler and an SVM
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1))
])

# Train the model on the training data
model.fit(X_train, y_train)
model

# Test the model on the test data
y_pred = model.predict(X_test)
y_pred

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy:.2f}')


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Test the model on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
report = classification_report(y_test, y_pred)
print(report)

# Plot confusion matrix
confusion = confusion_matrix(y_test, y_pred)
plt.imshow(confusion, cmap='binary', interpolation='none')
plt.colorbar()
plt.xlabel('true label')
plt.ylabel('predicted label')
plt

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

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred_train)
print(report)