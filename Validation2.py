# Step 7

import matplotlib.pyplot as plt
import pandas as pd
import ast
import math
import joblib
import numpy as np
import re
from sklearn import preprocessing, metrics
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing, metrics, svm
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

dp = pd.read_csv('Grid_search.csv')
dpx = dp['Best params']
df = pd.read_csv('Train and test.csv')
dv = pd.read_csv('Val_2.csv')
df = df.sample(frac=1, random_state=36)

results_df = []
feature_s = ['Mpo', 'Dar', 'Mmbe', 'Dgnve', 'Aose']
X = df[feature_s].values
y = df['target'].values

X_val = dv[feature_s].values
y_val = dv['target'].values


param_svrrbf = ast.literal_eval(dpx.iloc[2])
param_gbdt = ast.literal_eval(dpx.iloc[5])
param_LightGBM = ast.literal_eval(dpx.iloc[4])
param_AdaBoost = ast.literal_eval(dpx.iloc[6])
param_RandomForest = ast.literal_eval(dpx.iloc[0])
param_XGBoost = ast.literal_eval(dpx.iloc[1])
param_svrlinear = ast.literal_eval(dpx.iloc[3])
models = {
    'SVR.rbf': svm.SVR(kernel='rbf').set_params(**param_svrrbf),
    'GBDT': GradientBoostingRegressor(random_state=3407).set_params(**param_gbdt),
    'LightGBM': LGBMRegressor(seed=3407).set_params(**param_LightGBM),
    'AdaBoost': AdaBoostRegressor(random_state=3407).set_params(**param_AdaBoost),
    'Random Forest': RandomForestRegressor(random_state=3407).set_params(**param_RandomForest),
    'XGBoost': XGBRegressor(seed=3407).set_params(**param_XGBoost),
    'SVR.linear': svm.SVR(kernel='linear').set_params(**param_svrlinear)
}


standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)
X_val = standard_scaler.transform(X_val)

predictions = {}
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X_val)
    predictions[name] = y_pred
    metrics = {}

for name, y_pred in predictions.items():
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    metrics[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}

df = pd.DataFrame(metrics).transpose()
df.to_csv('model_metrics.csv')

