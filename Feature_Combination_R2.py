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
import time

start = time.time()

df = pd.read_csv('Train and test.csv')
features = ['Mpo', 'Aare', 'Dln', 'Mir', 'Dar', 'Mmbe', 'Aose', 'Dgnve', 'Wmn']
X = df[features]
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
                                     scoring='r2')
            score = np.mean(scores)

            results.append((len(sub), sub, score))

    result = pd.DataFrame(results,
                          columns=['length', 'subset', 'X_mean_R2'])
    min_rmse_row = result[result['X_mean_R2'] == result['X_mean_R2'].max()]
    best_subset = min_rmse_row['subset'].tolist()[0]
    min_r2 = min_rmse_row['X_mean_R2'].iloc[0]

    min_R2 = result.groupby("length")["X_mean_R2"].max()
    print(min_R2)
    min_result = pd.DataFrame({"x_min": min_R2}).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(result['length'], result['X_mean_R2'], label='subset',marker='o', edgecolors='blue',alpha = 0.5, facecolors='none')

    ax.plot(min_result['length'], min_result['x_min'], label='best_performance',color = 'red')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('R$^{{2}}$')

    ax.set_xticks(range(1, len(features)+1))
    ax.legend()

    plt.savefig('Features_Combination_R2.png', dpi=400)
    plt.show()

    result.to_csv('results_R2.csv', index=False)
    print("Best subset: ", best_subset)
    print("Length of best subset: ", len(best_subset))
    print("The maximum：", min_r2)


lgb = LGBMRegressor(seed=3407)
feature_combination(lgb, features, 'LGBM')

end = time.time()
runTime = end - start
print("Runtime(Seconds): ", runTime)
print("Runtime(Minutes): ", int(runTime/60))