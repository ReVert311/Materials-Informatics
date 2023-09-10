# Step 5

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import ast
import math
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn import svm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('Train and test.csv')
dp = pd.read_csv('Grid_search.csv')
dpx = dp['Best params']
df = df.sample(frac=1, random_state=3407)

features = ['Mpo', 'Dar', 'Mmbe', 'Aose', 'Dgnve']
x_train = df[features].values
y_train = df['target'].values


kf = KFold(n_splits=10, shuffle=True, random_state=2375)

models = {
    'S.R': svm.SVR(kernel='rbf').set_params(**ast.literal_eval(dpx.iloc[2])),
    'GBDT': GradientBoostingRegressor(random_state=3407).set_params(**ast.literal_eval(dpx.iloc[5])),
    'LGBM': LGBMRegressor(seed=3407).set_params(**ast.literal_eval(dpx.iloc[4])),
    'Ada': AdaBoostRegressor(random_state=3407).set_params(**ast.literal_eval(dpx.iloc[6])),
    'RF': RandomForestRegressor(random_state=3407).set_params(**ast.literal_eval(dpx.iloc[0])),
    'XGB': XGBRegressor(seed=3407).set_params(**ast.literal_eval(dpx.iloc[1])),
    'S.L': svm.SVR(kernel='linear').set_params(**ast.literal_eval(dpx.iloc[3]))
}

standard_scaler = preprocessing.StandardScaler()
x_train = standard_scaler.fit_transform(x_train)

ds = []
for name, model in models.items():
    y_pred = cross_val_predict(model, x_train, y_train, cv=kf)

    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    print(f'{name} Model score——r2: {r2}, mae: {mae},rmse: {rmse}')
    ds.append((name, r2, rmse, mae))

    plt.scatter(y_train, y_pred, s=30, alpha=0.5, edgecolors='C8', linewidths=1)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--', color='gray')
    plt.text(10, 28, 'R$^{{2}}$ = {:.4f}'.format(r2), fontsize=15)

    plt.xlabel('Experimental Values')
    plt.ylabel('Predicted Values')
    plt.title(name)
    plt.savefig('cvp_of_{}.png'.format(name))
    plt.show()

dsc = pd.DataFrame(ds, columns=['model', 'R2', 'RMSE', 'MAE'])
dsc.to_csv('cvp_models.csv')
metrics = ['R2', 'RMSE', 'MAE']
sns.set(style="whitegrid")
plt.figure(figsize=(10, 3.65))
for i, metric in enumerate(metrics):

    scores = dsc[metric].values
    algorithms = dsc['model'].values

    plt.subplot(1, 3, i + 1)
    plt.bar(algorithms, scores)

    plt.grid(False)
    plt.xticks(rotation=45)

    if i == 0:
        plt.ylabel('R$^{{2}}$')
    elif i == 1:
        plt.ylabel('RMSE')
    else:
        plt.ylabel('MAE')


plt.subplots_adjust(wspace=0.5)
plt.tight_layout()
plt.savefig('cvp_all_metrics.png')
plt.show()

