import pandas as pd
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import xgboost
from sklearn.model_selection import train_test_split


data = pd.read_csv('final_data.csv')
red_data = pd.read_csv('red_win.csv')
blue_data = pd.read_csv('blue_win.csv')

data.drop(columns='Unnamed: 0', inplace=True)
red_data.drop(columns='Unnamed: 0', inplace=True)
blue_data.drop(columns='Unnamed: 0', inplace=True)

data.winner = data['winner'].replace("Red", 0)
data.winner = data['winner'].replace("Blue", 1)
data




plt.figure(figsize=(8,5))
redblue = ["red", "blue"]
ax = sns.countplot(x='winner', data=data, palette=redblue)

plt.figure(figsize=(12,8))
redblue = ["red", "blue"]
ax = sns.countplot(y='weight_class', hue='winner', data=data, palette=redblue)


heavy_data = data[data['weight_class'] == 'Heavyweight']
feather_data = data[data['weight_class'] == 'Featherweight']


red_data.describe()


blue_data.describe()


plt.figure(figsize=(12,5))
sns.distplot( red_data["R_age"] , color="red", label="Age of Winner in Red Corner")
sns.distplot( blue_data["B_age"] , color="skyblue", label="Age of Winner in Blue Corner")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
sns.distplot( red_data["R_Height_cms"] , color="red", label="Age of Winner in Red Corner")
sns.distplot( blue_data["B_Height_cms"] , color="skyblue", label="Age of Winner in Blue Corner")
plt.legend()
plt.show()


plt.figure(figsize=(12,5))
ax = sns.countplot(x='R_age', data=red_data)

plt.figure(figsize=(12,5))
ax = sns.countplot(x='B_age', data=blue_data)



corr = data.loc[:, data.dtypes == 'float64'].corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, vmin=0, vmax=1)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


corrred = red_data.loc[:, red_data.dtypes == 'float64'].corr()
ax = sns.heatmap(corrred, xticklabels=corrred.columns, yticklabels=corrred.columns, annot=True, vmin=0, vmax=1)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

corrblue = blue_data.loc[:, blue_data.dtypes == 'float64'].corr()
ax = sns.heatmap(corrblue, xticklabels=corrblue.columns, yticklabels=corrblue.columns, annot=True, vmin=0, vmax=1)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


data.isnull().sum()



y = data['winner']
X = data.drop(columns=['winner', 'R_fighter', 'B_fighter', 'weight_class'])
weightdummies = pd.get_dummies(data.weight_class, drop_first=True)
X = pd.concat([weightdummies, X], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)


# y = feather_data['winner']
# X = feather_data.drop(columns=['winner', 'R_fighter', 'B_fighter', 'weight_class'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)



clf = XGBClassifier()
clf.fit(X_train, y_train)
training_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)

# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))

param_grid = {
    'learning_rate': [0.1, 0.2, 0.3, 0.5],
    'max_depth': [4, 5, 6],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.5, 0.7],
    'n_estimators': [1, 10, 100, 1000]
}


grid_clf = GridSearchCV(clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
grid_clf.fit(X_train, y_train)

best_parameters = grid_clf.best_params_
best_parameters

training_preds = grid_clf.predict(X_train)
test_preds = grid_clf.predict(X_test)
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('')
print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))


clf = XGBClassifier(learning_rate=0.1, max_depth=7, min_child_weight=1, n_estimators=10, subsample=0.6)
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy of %s is %s"%(clf, acc))
print("")
print("Precision of %s is %s"%(clf, prec))
print("")
print("Recall of %s is %s"%(clf, recall))
print("")
print("F1 Score of %s is %s"%(clf, f1))
print("")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")
print("")





clf = RandomForestClassifier()
clf.fit(X_train, y_train)
training_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)

# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))


param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [1, 10, 100, 1000],
}


grid_clf = GridSearchCV(clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
grid_clf.fit(X_train, y_train)

best_parameters = grid_clf.best_params_
best_parameters

training_preds = grid_clf.predict(X_train)
test_preds = grid_clf.predict(X_test)
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('')
print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))



clf = RandomForestClassifier(max_features='auto', n_estimators=10)
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy of %s is %s"%(clf, acc))
print("")
print("Precision of %s is %s"%(clf, prec))
print("")
print("Recall of %s is %s"%(clf, recall))
print("")
print("F1 Score of %s is %s"%(clf, f1))
print("")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")
print("")



C=0.8, max_iter=500

clf = LogisticRegression()
clf.fit(X_train, y_train)
training_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)


# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 200, 300]
}


grid_clf = GridSearchCV(clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
grid_clf.fit(X_train, y_train)

best_parameters = grid_clf.best_params_
best_parameters

training_preds = grid_clf.predict(X_train)
test_preds = grid_clf.predict(X_test)
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('')
print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))



clf = LogisticRegression(C=0.001, penalty='l2', max_iter=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy of %s is %s"%(clf, acc))
print("")
print("Precision of %s is %s"%(clf, prec))
print("")
print("Recall of %s is %s"%(clf, recall))
print("")
print("F1 Score of %s is %s"%(clf, f1))
print("")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")
print("")
 
