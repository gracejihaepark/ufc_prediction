# UFC Predictions
Machine learning algorithms to predict the winners of UFC fights

## Goals
- Predict the winners of UFC fights by the corners they fight out of, and their physical attributes
- Use different classification models to find the best model for prediction
  - Random Forest, Logistic Regression, XGBoost

## Data
- Dataset from Kaggle
  - Eliminated all features that were related to strikes attempted and landed to predict the winner before the fight begins
  - Fill in NaNs with medians of column

## Exploratory Data Analysis
- There was a much higher percentage of winners coming from the red corner
  - Realized this was because the red corner was usually for the fighter thought to win, and the blue corner was usually the underdog of the fight
![alt text](https://github.com/gracejihaepark/ufc_prediction/blob/master/readme%20images/Screen%20Shot%202020-01-31%20at%2010.56.21%20AM.png?raw=true)(https://github.com/gracejihaepark/ufc_prediction/blob/master/readme%20images/Screen%20Shot%202020-01-31%20at%2010.56.32%20AM.png?raw=true)

- The heights, reaches, and age of fighters seem to benefit the blue corner fighters more though
  - Since the blue corner is the underdog of the fight, they are most likely newer to UFC, which in turn can make them younger, and although heights and reaches are better, they probably don't have the experience yet compared to seasoned fighters in the red corner
![alt text](https://github.com/gracejihaepark/ufc_prediction/blob/master/readme%20images/Screen%20Shot%202020-01-31%20at%2012.02.09%20PM.png?raw=true)(https://github.com/gracejihaepark/ufc_prediction/blob/master/readme%20images/Screen%20Shot%202020-01-31%20at%2012.03.14%20PM.png?raw=true)(https://github.com/gracejihaepark/ufc_prediction/blob/master/readme%20images/Screen%20Shot%202020-01-31%20at%2012.03.24%20PM.png?raw=true)
