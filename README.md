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
