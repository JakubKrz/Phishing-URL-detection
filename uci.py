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
print(phiusiil_phishing_url_website.) 

#TO DO: Change categorical columns into encoded(numeric values) columns
#1 - one-hot encoding 2 - label encoding 3 - ordinal encoding

#finding categorical columns
categorical_columns = []

for index, row in phiusiil_phishing_url_website.variables.iterrows():
    if row['type'] == 'Categorical':
        categorical_columns.append(row['name'])

label_encoder = LabelEncoder()
#encode every categorical column, but not FILENAME (first one)
for x in categorical_columns[1:]:
    X.isetitem(X.columns.get_loc(x), label_encoder.fit_transform(X[x]))
    #X.loc[:, x] = label_encoder.fit_transform(X.loc[:, x])
    #label_encoder.fit_transform(X[X.loc[:, x]])

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create Decision Tree classifer object
DTC = DecisionTreeClassifier()
# Train Decision Tree Classifer
DTC = DTC.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = DTC.predict(X_test)
#Model accuracy
print(metrics.accuracy_score(y_test, y_pred))


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# visualization of DTC
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(DTC, feature_names=X_train.columns, class_names=['phishing', 'not phishing'], filled=True)
plt.show()

#Drawing nonzero feature inmportance
feature_importance = DTC.feature_importances_
feature_names = X_train.columns
nonzero_feature_importance = feature_importance[feature_importance > 0]
nonzero_feature_names = feature_names[feature_importance > 0]

plt.figure(figsize=(10, 6), dpi=500)
plt.barh(nonzero_feature_names, nonzero_feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance in Decision Tree Classifier')
plt.show()