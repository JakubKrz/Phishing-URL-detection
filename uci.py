from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# fetch dataset 
phiusiil_phishing_url_website = fetch_ucirepo(id=967)

# data (as pandas dataframes)
X = phiusiil_phishing_url_website.data.features
y = phiusiil_phishing_url_website.data.targets

# metadata 
print(phiusiil_phishing_url_website.metadata)   
# variable information 
print(phiusiil_phishing_url_website.variables) 

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#TO DO: Change categorical columns into encoded(numeric values) columns
#1 - one-hot encoding 2 - label encoding 3- ordinal encoding

#finding categorical columns
categorical_indexes = []
for idx, column in enumerate(phiusiil_phishing_url_website.variables.columns):
    if phiusiil_phishing_url_website.variables.loc[idx, 'type'] == 'Categorical':
        categorical_indexes.append(idx)

# Create Decision Tree classifer object
DTC = DecisionTreeClassifier()
# Train Decision Tree Classifer
DTC = DTC.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = DTC.predict(X_test)
#Model accuracy
metrics.accuracy_score(y_test, y_pred)

#TO DO: add visualization of decision tree (use graphviz and pydotplus?)