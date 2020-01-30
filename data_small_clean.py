import pandas as pd
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data_small.csv')
data = data.drop(columns='Unnamed: 0')

data


data.fillna(data.mean(), inplace=True)

data.isnull().sum()

data.R_Stance.fillna('N/A', inplace=True)
data.B_Stance.fillna('N/A/', inplace=True)

data.isnull().sum()


# fig = plt.figure(figsize = (15,20))
# data.hist()


red_win = data[data['winner'] == 'Red']
red_win


blue_win = data[data['winner'] == 'Blue']
blue_win

red_win.describe()
blue_win.describe()




plt.figure(figsize=(10,8))
redblue = ["red", "blue"]
ax = sns.countplot(x='winner', data=data, palette=redblue)

plt.figure(figsize=(15,8))
redblue = ["red", "blue"]
ax = sns.countplot(y='weight_class', hue='winner', data=data, palette=redblue)
