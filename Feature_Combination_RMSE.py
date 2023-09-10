# Step 3

from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import ast
from sklearn.model_selection import KFold
import joblib
import numpy as np
import re
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_validate


df = pd.read_csv('Train and test.csv')

features = ['Mpo', 'Aare', 'Dln', 'Mir', 'Dar', 'Mmbe', 'Aose', 'Dgnve', 'Wmn']

y = df['target']

standard_scaler = preprocessing.StandardScaler()

def feature_combination(model, fs, str):
    best_subset = None
    results = []
    for i in range(1, len(fs) + 1):
        for subset in combinations(fs, i):
            sub = list(subset)
            x = df[sub]
            x = standard_scaler.fit_transform(x)

            scores = cross_val_score(model, x, y, cv=KFold(n_splits=10, shuffle=True, random_state=4251), n_jobs=-1,
                                     scoring='neg_root_mean_squared_error')
            score = np.mean(-scores)

            results.append((len(sub), sub, score))

    result = pd.DataFrame(results,
                          columns=['length', 'subset', 'X_mean_RMSE'])
    min_rmse_row = result[result['X_mean_RMSE'] == result['X_mean_RMSE'].min()]
    best_subset = min_rmse_row['subset'].tolist()[0]
    min_rmse = min_rmse_row['X_mean_RMSE'].iloc[0]

    min_RMSE = result.groupby("length")["X_mean_RMSE"].min()
    print(min_RMSE)

    min_result = pd.DataFrame({"x_min": min_RMSE}).reset_index()


    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(result['length'], result['X_mean_RMSE'], label='subset',marker='o', edgecolors='blue',alpha = 0.5, facecolors='none')
    ax.plot(min_result['length'], min_result['x_min'], label='best_performance',color = 'red')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('RMSE')
    ax.set_xticks(range(1, len(features)+1))
    ax.legend()
    plt.savefig('Features_Combination_RMSE.png', dpi=300)
    plt.show()

    result.to_csv('results_RMSE.csv', index=False)
    print("Best subset: ", best_subset)
    print("Length of best subset: ", len(best_subset))
    print("The minimum：", min_rmse)


strr = 'Spinel'
m = LGBMRegressor(seed=3407)
feature_combination(m, features, strr)
