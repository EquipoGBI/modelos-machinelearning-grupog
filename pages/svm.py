import yfinance as yf
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Get stock data for BVN from yahoo finance
bvn = yf.Ticker("BVN").history(period="max")

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

# Plot feature importance
model.named_steps['svm'].plot_importance(precision=3)
model