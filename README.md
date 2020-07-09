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
![alt text](https://github.com/gracejihaepark/ufc_prediction/blob/master/images/winner%20by%20corner%20colors.png?raw=true)

- The height, reach, and age of fighters seem to benefit the blue corner fighters more though
  - Since the blue corner is the underdog of the fight, they are most likely newer to UFC, which in turn can make them younger, and although height and reach are better, they probably don't have the experience yet compared to seasoned fighters in the red corner
![alt text](https://github.com/gracejihaepark/ufc_prediction/blob/master/images/attributes%20of%20fighers%20by%20corner%20colors.png?raw=true)

- The height, reach, and age of just the winners are pretty similar for both corners, and the peaks of these curves are more defined than compared with all fighers in the UFC
![alt text](https://github.com/gracejihaepark/ufc_prediction/blob/master/images/attributes%20of%20winners%20by%20corner%20colors.png?raw=true)

## Model
- Ran Random Forest, Logistic Regression, and XGBoost, running gridsearch for all three models to find the best parameters
- All three models were similar, being off by .01 between models
![alt text](https://github.com/gracejihaepark/ufc_prediction/blob/master/images/model.png?raw=true)

## Conclusions
- XGBoost was the best model for predicting UFC fight winners
- Accuracy was overall not very high, but UFC fights have many uncontrollable factors as well as unpredictability in “underdog” wins

## Future Work
- Would like to dive into winner and loser attributes more
- Is there home court (country) advantage?
