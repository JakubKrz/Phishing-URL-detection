from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np

phiusiil_phishing_url_website = fetch_ucirepo(id=967)

X = phiusiil_phishing_url_website.data.features.copy()
y = phiusiil_phishing_url_website.data.targets

y = y.values.ravel() #flatten to 1d

categorical_columns = []

for index, row in phiusiil_phishing_url_website.variables.iterrows():
    if row['type'] == 'Categorical':
        categorical_columns.append(row['name'])

label_encoder = LabelEncoder()

for x in categorical_columns[1:]:
    X.isetitem(X.columns.get_loc(x), label_encoder.fit_transform(X[x]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

CLF = Perceptron(eta0=0.001)

classes = np.unique(y_train)
n_epochs = 10
batch_size = 32

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    # Shuffle data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train.iloc[indices]
    y_train = y_train[indices]

    # Split data and train
    for start in range(0, X_train.shape[0], batch_size):
        end = min(start + batch_size, X_train.shape[0])
        X_batch = X_train.iloc[start:end]
        y_batch = y_train[start:end]
        CLF.partial_fit(X_batch, y_batch, classes=classes)
    y_pred = CLF.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'\nAccuracy: {accuracy:.2f}')

print("\nClassification report:")
print(metrics.classification_report(y_test, y_pred))

print("\nComparing actual and predicted labels for the first 10 test samples:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
