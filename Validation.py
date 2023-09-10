# Step 6

import matplotlib.pyplot as plt
import pandas as pd
import ast
import joblib
import numpy as np
import re
from sklearn import preprocessing, metrics
from xgboost import XGBRegressor

model = XGBRegressor(seed=3407)

df = pd.read_csv('Train and test.csv')
dv = pd.read_csv('Val.csv')
df = df.sample(frac=1, random_state=36)

results_df = []
feature_s = ['Mpo', 'Dar', 'Mmbe', 'Dgnve', 'Aose']
X = df[feature_s].values
y = df['target'].values

X_val = dv[feature_s].values
y_val = dv['target'].values

standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)
model.fit(X, y)
y_vp = model.predict(X_val)

x = [0.01, 0.03, 0.05, 0.07, 0.09]
y_cm = [10.34, 10.32, 10.27, 10.21, 10.18]

plt.plot(x, y_val, label='Experimental', marker='o', linestyle='-')
plt.plot(x, y_vp, label='Machine Learning', marker='s', linestyle='--')
plt.plot(x, y_cm, label='C-M Equation', marker='^', linestyle=':')

plt.xlabel('X values')
plt.ylabel('ε${_r}$')
plt.title('Val of Mg${_2}$Ti${_1}$${_-}$${_x}$Al${_4}$${_/}$${_3}$${_x}$O${_4}$')
plt.xticks(x)
plt.ylim(9, 18)
plt.legend(loc='upper right')
plt.savefig('val.png')
plt.show()