# Step 4

import pandas as pd
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import time

start = time.time()

df = pd.read_csv('Train and test.csv')
df = df.sample(frac=1, random_state=3407)

features = ['Mpo', 'Dar', 'Mmbe', 'Aose', 'Dgnve']

x_train = df[features].values
y_train = df['target'].values

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x_train = min_max_scaler.fit_transform(x_train)

scoring = {
    'r2': make_scorer(r2_score),
    'mae': make_scorer(mean_absolute_error),
}


class ModelTuner:
    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def rf(self, param_dict):
        self.name = 'Random Forest'
        param_dict['random_state'] = 3407

        return RandomForestRegressor(**param_dict), {
            'n_estimators': [10, 20, 40, 60, 100],
            'max_depth': [2, 5, 8, 11],
        }

    def xgb(self, param_dict):
        self.name = 'XGBoost'
        param_dict['random_state'] = 3407

        return XGBRegressor(**param_dict), {
            'n_estimators': [10, 20, 40, 60, 100],
            'max_depth': [2, 5, 8, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.5],
        }

    def svrr(self, param_dict):
        self.name = 'svr.rbf'
        return svm.SVR(**param_dict), {
            'kernel': ['rbf'],
            'C': [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,32,64,128],
            'gamma': [0.1, 1, 10, 2.0, 4.0, 8.0, 16.0,32,64, 100],
            'epsilon': [0.01, 0.1, 1, 10]
        }

    def svrl(self, param_dict):
        self.name = 'svr.linear'
        return svm.SVR(**param_dict), {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1, 10]
        }

    def adab(self, param_dict):
        self.name = 'AdaBoost'
        param_dict['random_state'] = 3407

        return AdaBoostRegressor(**param_dict), {
            'n_estimators': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.1, 1, 10]
        }

    def lgbm(self, param_dict):
        self.name = 'LightGBM'
        param_dict['random_state'] = 3407

        return LGBMRegressor(**param_dict), {
            'max_depth': [-1, 3, 5, 7, 10, 13],
            'learning_rate': [0.01, 0.1, 1],
        }

    def gbdt(self, param_dict):
        self.name = 'GradientBoosting'
        param_dict['random_state'] = 3407
        return GradientBoostingRegressor(**param_dict), {
            'n_estimators': [20, 50, 100, 200],
            'learning_rate': [1, 0.5, 0.2, 0.1, 0.01],
            'max_depth': [3, 5, 7],
        }


    def grid_tuning(self, model_fn, param_space, x_train, y_train):
        sed = 2375
        model, space = model_fn(param_space)
        grid_search = GridSearchCV(
            model,
            space,
            scoring='neg_mean_squared_error',
            cv=KFold(n_splits=10, shuffle=True, random_state=sed),
            verbose=0,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)
        cv_result = grid_search.cv_results_
        cvr = pd.DataFrame.from_dict(cv_result, orient='index').T
        best_params = grid_search.best_params_
        model_cvs = model.set_params(**best_params)
        result_mae = cross_val_score(model_cvs, x_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=sed), scoring='neg_mean_absolute_error')
        result_mae = -result_mae
        result_r2 = cross_val_score(model_cvs, x_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=sed), scoring='r2')
        best_score = grid_search.best_score_
        return {'Model name': self.name,
                'Best params': best_params,
                'Best RMSE score': np.sqrt(-best_score),
                'R2 score': result_r2.mean(),
                'MAE score': result_mae.mean()
                }, cvr



model_tuner = ModelTuner()

dict1, cvr1 = model_tuner.grid_tuning(model_tuner.rf, {}, x_train, y_train)
print(dict1)
df1 = pd.DataFrame.from_dict(
    dict1,
    orient='index').T

dict2, cvr2 = model_tuner.grid_tuning(model_tuner.xgb, {}, x_train, y_train)
print(dict2)
df2 = pd.DataFrame.from_dict(
    dict2,
    orient='index').T

dict3, cvr3 = model_tuner.grid_tuning(model_tuner.svrr, {}, x_train, y_train)
print(dict3)
df3 = pd.DataFrame.from_dict(
    dict3,
    orient='index').T

dict4, cvr4 = model_tuner.grid_tuning(model_tuner.svrl, {}, x_train, y_train)
print(dict4)
df4 = pd.DataFrame.from_dict(
    dict4,
    orient='index').T

dict5, cvr5 = model_tuner.grid_tuning(model_tuner.lgbm, {}, x_train, y_train)
print(dict5)
df5 = pd.DataFrame.from_dict(
    dict5,
    orient='index').T

dict6, cvr6 = model_tuner.grid_tuning(model_tuner.gbdt, {}, x_train, y_train)
print(dict6)
df6 = pd.DataFrame.from_dict(
    dict6,
    orient='index').T

dict7, cvr7 = model_tuner.grid_tuning(model_tuner.adab, {}, x_train, y_train)
print(dict7)
df7 = pd.DataFrame.from_dict(
    dict7,
    orient='index').T
dc = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0, ignore_index=True)
dc.to_csv('Grid_search.csv')

data = {'RandomForest': cvr1, 'XGBoost': cvr2, 'svr.rbf': cvr3, 'svr.linear': cvr4, 'LightGBM': cvr5,
        'GradientBoosting': cvr6, 'AdaBoost': cvr7}

end = time.time()
runTime = end - start
print("Runtime(Minutes)：", int(runTime/60))