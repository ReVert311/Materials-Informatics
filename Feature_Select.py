# Step 1

import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

dfy = pd.read_csv('Train and test.csv')

feature = ['Aln', 'Aare', 'Aose', 'Daw', 'Dln', 'Dar', 'Dcor', 'Dare', 'Dgnve', 'Dip', 'Dpo', 'Dd', 'Rar', 'Rose', 'Xaw', 'Xln', 'Xcor', 'Xcrr', 'Xmv', 'Xd', 'Xf', 'Xnm', 'Xc', 'Nln', 'Ncrr', 'Ngnve', 'Nip', 'Nd', 'Nc', 'Maw', 'Mir', 'Mmbe', 'Mare', 'Mnve', 'Mgnve', 'Mip', 'Mpo', 'Md', 'Mmn', 'Saw', 'Sln', 'Scor', 'Sir', 'Scrr', 'Smv', 'Snve', 'Sgnve', 'Spo', 'Sd', 'Sf', 'Sc', 'Sr', 'War', 'Wcrr', 'Wpe', 'Ware', 'Wose', 'Wip', 'Wpo', 'Wm', 'Wmn', 'Wc']
features = dfy[feature]
y = dfy['target']

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
feature_final = min_max_scaler.fit_transform(features)

feature_transform = pd.DataFrame(feature_final, columns=features.columns)

selector = VarianceThreshold(threshold=0.015)
result_select = selector.fit_transform(feature_transform)

var = selector.variances_


result_support = selector.get_support(indices=True)
print("result support:", result_support)
select_list = result_support
select11 = feature_transform.iloc[:, select_list]
selected_features = select11.columns.tolist()

all_features = feature_transform.columns.tolist()
unselected_features = [feature for feature in all_features if feature not in selected_features]

mi_scores = mutual_info_regression(select11, y)
features_dict = {select11.columns[i]: mi_scores[i] for i in range(select11.shape[1])}
sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
count = sum(1 for feature, score in sorted_features if score >= 0.99)

top_features = sorted_features[:40]
labels = [x[0] for x in top_features]
scores = [x[1] for x in top_features]

theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

max_score = max(scores)
normalized_scores = [score / max_score for score in scores]


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, )
bars = ax.bar(theta, normalized_scores, width=0.4, align='center', alpha=0.7)


ax.set_xticks(theta)
ax.set_xticklabels(labels, fontsize=8)


cmap = plt.get_cmap('coolwarm')
norm = plt.Normalize(vmin=0, vmax=max_score)
colors = cmap(norm(scores))
for bar, color in zip(bars, colors):
    bar.set_color(color)


plt.title('Mutual Information Scores for Top 40 Features')
plt.subplots_adjust(top=0.8)
plt.subplots_adjust(right=0.8)
cax = plt.colorbar(mappable=cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', pad=0.15, shrink=0.8)
ax.tick_params(axis='y', labelsize=8)
cax.set_label('Normalized Score')
plt.tight_layout()
plt.savefig('Mutual Information.png')
plt.show()

sorted_feature_names = [feature for feature, score in sorted_features]

new_df = features[sorted_feature_names]



pear_scores = select11.corrwith(y)
pear_dict = {select11.columns[i]: pear_scores[i] for i in range(select11.shape[1])}


corr = new_df.corr(method='pearson')


np.fill_diagonal(corr.values, 0)

filtered_features = []


for feature in corr.columns:
    flag = True
    for corr_value in corr[feature]:
        if isinstance(corr_value, str):
            raise ValueError(
                "Correlation value {} for feature {} does not satisfy the filtering condition.".format(corr_value,
                                                                                                       feature))
        if abs(corr_value) >= 0.85:
            flag = False
            break
    if flag:
        filtered_features.append(feature)

print("Features without high correlation: ", filtered_features)

upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
abs_threshold = 0.85
corr_pairs = [(i, j) for i in upper_tri.index for j in upper_tri.columns if abs(upper_tri.loc[i, j]) >= abs_threshold]
cp = copy.deepcopy(corr_pairs)

new_features = []
to_be_revised_features = []

for pair in corr_pairs:
    if (pair[0] in [i[0] for i in sorted_features][:count]) and (pair[1] in [i[0] for i in sorted_features][:count]):
        continue
    else:
        score1 = [i[1] for i in sorted_features if i[0] == pair[0]][0]
        score2 = [i[1] for i in sorted_features if i[0] == pair[1]][0]
        if score1 < score2:
            new_features.append(pair[0])
        if score1 > score2:
            new_features.append(pair[1])
        else:
            to_be_revised_features.append(pair)

new_features = list(set(new_features))
to_remove = []
for i in corr_pairs:
    for feat in new_features:
        if feat in i:
            to_remove.append(i)
            break
for i in to_remove:
    if i in corr_pairs:
        corr_pairs.remove(i)

corr_pairs = [pair for pair in corr_pairs if pair not in to_be_revised_features]

to_remove_r = []
for i in to_be_revised_features:
    for feat in new_features:
        if feat in i:
            to_remove_r.append(i)
            break
for i in to_remove_r:
    if i in to_be_revised_features:
        to_be_revised_features.remove(i)


unique_features = []
for feature in set(new_features):
    unique_features.append(feature)


selected_final_features = [feature for feature in selected_features if feature not in unique_features]

print("To delete: ", unique_features)
print("Got left: ", corr_pairs)
print("Revise again: ", to_be_revised_features)
print("Got selected: ", selected_final_features)
print(len(selected_final_features))

fig, ax = plt.subplots(figsize=(15, 12))
mask_ut = np.triu(np.ones_like(corr, dtype=bool))
heatmap = sns.heatmap(corr, annot=False, cbar=False, vmax=1, vmin=-1, linewidths=0.5, linecolor='white',
                      xticklabels=True, yticklabels=True, square=True, cmap="RdBu")
heatmapcb = heatmap.figure.colorbar(heatmap.collections[0])
heatmapcb.ax.tick_params(labelsize=24)
heatmapcb.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig('Pearson', dpi=500)
plt.show()
