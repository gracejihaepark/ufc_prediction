import pandas as pd
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)



data_all = pd.read_csv('data.csv')
data_all
data_all.shape

data = data_all[data_all.date.str.contains('2010|2011|2012|2013|2014|2015|2016|2017|2018|2019')]
data
data.isnull().sum()

data = data.drop(columns=['B_draw', 'R_draw'])
data = data.rename(columns={"Winner": "winner", "Referee": "referee"})
data.winner.unique()
data = data[~data.winner.str.contains('Draw')]
data
data.isnull().sum()


data.winner.unique()


data.winner.value_counts().plot(kind='bar', rot=0)
data.winner.value_counts()

data.weight_class.value_counts()

data.R_Stance.value_counts()
data.B_Stance.value_counts()


data_small = data[['R_fighter', 'B_fighter', 'winner', 'weight_class', 'no_of_rounds', 'R_Stance', 'R_age', 'R_Height_cms', 'R_Reach_cms', 'B_Stance', 'B_age', 'B_Height_cms', 'B_Reach_cms']]


data_small.isnull().sum()
data_small.to_csv('data_small.csv')
