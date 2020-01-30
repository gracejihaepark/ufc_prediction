import pandas as pd
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)


data = pd.read_csv('data_small.csv')
data = data.drop(columns=['Unnamed: 0', 'R_Stance', 'B_Stance'])

data


data.isnull().sum()

data = data.fillna(data.median())


data.isnull().sum()

data.to_csv('final_data.csv')

red_win = data[data['winner'] == 'Red']
red_win.to_csv('red_win.csv')


blue_win = data[data['winner'] == 'Blue']
blue_win.to_csv('blue_win.csv')

red_win.describe()
blue_win.describe()
