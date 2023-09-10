# Step 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold


def calculate_feature_importance(data, target, algorithms, st='gini'):
    df_feature_importance = pd.DataFrame(index=data.columns)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data = min_max_scaler.fit_transform(data)

    for algorithm in algorithms:
        if algorithm == 'GBDT':
            forest = GradientBoostingRegressor(random_state=3407)
        elif algorithm == 'RandomForest':
            forest = RandomForestRegressor(random_state=3407)
        elif algorithm == 'XGBoost':
            forest = XGBRegressor(random_state=3407)
        forest.fit(data, target)

        if st == 'gini':
            importances = forest.feature_importances_


        df_feature_importance[algorithm] = importances

    df_feature_importance['Total Importance'] = df_feature_importance.sum(axis=1)
    df_feature_importance_sorted = df_feature_importance.sort_values(by='Total Importance', ascending=False)
    feature_list = df_feature_importance_sorted.head(25).index.tolist()
    df_feature_importance_sorted = df_feature_importance_sorted.drop('Total Importance', axis=1)

    df_feature_importance_sorted.head(30).plot(kind='barh', stacked=True,
                                               title='Feature Importances of All Descriptors', legend='best',
                                               figsize=(10, 6))
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Descriptors', fontsize=12)
    plt.savefig('Feature Importances.png', dpi=400)
    plt.show()
    print(feature_list[:9])


df = pd.read_csv('Train and test.csv')

y = df['target']
algorithms = ['XGBoost', 'RandomForest', 'GBDT']
feature = ['Aln', 'Aare', 'Aose', 'Daw', 'Dln', 'Dar', 'Dcor', 'Dare', 'Dgnve', 'Dip', 'Dpo', 'Dd', 'Rar', 'Rose', 'Xaw', 'Xln', 'Xcor', 'Xcrr', 'Xmv', 'Xd', 'Xf', 'Xnm', 'Xc', 'Nln', 'Ncrr', 'Ngnve', 'Nip', 'Nd', 'Nc', 'Maw', 'Mir', 'Mmbe', 'Mare', 'Mnve', 'Mgnve', 'Mip', 'Mpo', 'Md', 'Mmn', 'Saw', 'Sln', 'Scor', 'Sir', 'Scrr', 'Smv', 'Snve', 'Sgnve', 'Spo', 'Sd', 'Sf', 'Sc', 'Sr', 'War', 'Wcrr', 'Wpe', 'Ware', 'Wose', 'Wip', 'Wpo', 'Wm', 'Wmn', 'Wc']
reasu = []
features = df[feature]
calculate_feature_importance(features, y, algorithms, 'gini')
