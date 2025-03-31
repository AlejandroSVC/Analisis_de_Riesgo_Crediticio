# Estimating client credit risk
## Using xgBoost Machine Learning Classification with Python

![Credit_risk](docs/assets/images/Banner_credit_risk.jpg)

### XGBOOST Machine Learning model

XGBoost stands for eXtreme Gradient Boosting. It is focused on
computational speed and model performance. 

### PYTHON CODE:

### Set working directory and load data
```
import os

import pandas as pd

os.chdir('dir')

df = pd.read_csv('data.csv')

df.info()
```
### Import libraries
```
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
```
### 1. Separate features and target
```
X = data.drop('LoanApproved', axis=1)
y = data['LoanApproved']
```
### Split data into train and test sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 2. Define objective function for Optuna
```
from sklearn.metrics import root_mean_squared_error

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    return rmse
```
### 3. Run Optuna optimization
```
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
```
### 4. Train final model with best hyperparameters
```
best_params = study.best_params
best_params['objective'] = 'reg:squarederror'
best_params['random_state'] = 42

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
```
### 5. Evaluate the model
```
y_pred = final_model.predict(X_test)

print("\nEvaluation Metrics:")
```
### Calculate evaluation metrics
```
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)
```
### Print the evaluation metrics
```
print(f"MAPE, mean absolute percentage error:", round(mape,5))
print(f"MSE, Mean squared error:", round(mse,5))
print(f"RMSE, Root mean squared error:", round(rmse,5))
print(f"MAE, Mean absolute error:", round(mae,5))
print(f"R2, R-squared:", round(r2,5))
print(f"Explained variance:", round(explained_var,5))
```
OUTPUT

Evaluation Metrics:

MAPE, mean absolute percentage error: 45035996273704.97

MSE, Mean squared error: 0.025

RMSE, Root mean squared error: 0.15811

MAE, Mean absolute error: 0.025

R2, R-squared: 0.86294

Explained variance: 0.86308

### PLOTS

### 6. Feature importance
```
feature_importance = final_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.xticks(range(X.shape[1]), sorted_idx)
plt.xlabel('Feature index')
plt.ylabel('Feature importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
```
### 7. Optimization history plot
```
optuna.visualization.plot_optimization_history(study).show()
```
### 8. Parameter importance plot
```
optuna.visualization.plot_param_importances(study).show()
```
