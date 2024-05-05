from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# fetch dataset 
phiusiil_phishing_url_website = fetch_ucirepo(id=967)

# data (as pandas dataframes)
#X = phiusiil_phishing_url_website.data.features
X = phiusiil_phishing_url_website.data.features.copy()
y = phiusiil_phishing_url_website.data.targets

# metadata 
print(phiusiil_phishing_url_website.metadata)   
# variable information 
print(phiusiil_phishing_url_website.variables) 

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#TO DO: Change categorical columns into encoded(numeric values) columns
#1 - one-hot encoding 2 - label encoding 3 - ordinal encoding

# LabelEncoder()

#finding categorical columns
categorical_indexes = []
for idx, column in enumerate(phiusiil_phishing_url_website.variables.columns):
    if phiusiil_phishing_url_website.variables.loc[idx, 'type'] == 'Categorical':
        categorical_indexes.append(idx)

categorical_columns = phiusiil_phishing_url_website.variables.loc[categorical_indexes, 'name'].tolist()

label_encoder = LabelEncoder()
for x in categorical_columns:
    X.loc[:, x] = label_encoder.fit_transform(X.loc[:, x])
#moze zrobic dla X_test i X_train, z jakiehgos powodu nie robi dla kolumny title

# Create Decision Tree classifer object
DTC = DecisionTreeClassifier()
# Train Decision Tree Classifer
DTC = DTC.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = DTC.predict(X_test)
#Model accuracy
metrics.accuracy_score(y_test, y_pred)

#TO DO: add visualization of decision tree (use graphviz and pydotplus?)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Narysuj drzewo decyzyjne
plt.figure(figsize=(20,10))
plot_tree(DTC, feature_names=X_train.columns, class_names=['phishing', 'not phishing'], filled=True)
plt.show()