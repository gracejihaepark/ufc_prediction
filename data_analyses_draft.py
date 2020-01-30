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

data

heavy_data = data[data['weight_class'] == 'Heavyweight']
feather_data = data[data['weight_class'] == 'Featherweight']


red_data.describe()


blue_data.describe()

plt.figure(figsize=(12,5))
ax = sns.countplot(x='R_age', data=red_data)

plt.figure(figsize=(12,5))
ax = sns.countplot(x='B_age', data=blue_data)



corr = data.loc[:, data.dtypes == 'float64'].corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


corrred = red_data.loc[:, red_data.dtypes == 'float64'].corr()
ax = sns.heatmap(corrred, xticklabels=corrred.columns, yticklabels=corrred.columns, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

corrblue = blue_data.loc[:, blue_data.dtypes == 'float64'].corr()
ax = sns.heatmap(corrblue, xticklabels=corrblue.columns, yticklabels=corrblue.columns, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


data.isnull().sum()



y = data['winner']
X = data.drop(columns=['winner', 'R_fighter', 'B_fighter', 'weight_class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

weightdummies = pd.get_dummies(data.weight_class, drop_first=True)
X = pd.concat([weightdummies, X], axis=1)

classifiers = []

model1 = xgboost.XGBClassifier()
classifiers.append(model1)

model2 = svm.SVC()
classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()
classifiers.append(model3)

model4 = RandomForestClassifier()
classifiers.append(model4)

model5 = LogisticRegression(C=0.8, max_iter=500)
classifiers.append(model5)


for clf in classifiers:
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
