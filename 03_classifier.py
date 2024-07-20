# %% md
# # 3. Classifier for the breast cancer patients based on PAM50 subtypes


# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scikit_posthocs as sp
import shap
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import friedmanchisquare, spearmanr
from sklearn.ensemble import (AdaBoostClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import (GridSearchCV, RepeatedKFold,
                                     cross_val_score, train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def save_fig(fig_, savepath):
    fig_.update_layout(dragmode='pan', margin=dict(l=0, r=0, t=30, b=30))
    fig_.write_html(savepath, config={'scrollZoom': True, 'displaylogo': False})


pio.templates.default = 'simple_white'

# %%
dataset_full = pd.read_csv('data/processed/filtered_dataset.csv', index_col=0)
metadata_full = pd.read_csv('data/processed/metadata.csv', index_col=0)

metadata = metadata_full.dropna(subset='PAM50').reset_index(drop=True)
dataset = dataset_full.loc[:, metadata['submitter_id.samples']]

degs_subset = pd.read_csv('data/processed/degs_subset.csv', index_col=0)


# %% md
# ## All genes

# %%
X = dataset.T
y = metadata['PAM50']

# %%
# split the data into train test and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# %%
classifiers_all = {
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=10000, random_state=42)),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'Neural Network': make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(200, 200, 100),
        max_iter=1000,
        early_stopping=True,
        random_state=42))
}

# %%
# train and evaluate classifiers
results = {}
for name, clf in (pbar := tqdm(classifiers_all.items())):
    pbar.set_description(f'fitting {name}')
    clf.fit(X_train, y_train)
    pbar.set_description(f'predicting with {name}')
    y_pred = clf.predict(X_val)
    pbar.set_description(f'evaluating {name}')
    y_proba = (clf.predict_proba(X_val)
               if hasattr(clf, "predict_proba") else clf.decision_function(X_val))

    results[name] = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_val, y_pred, average='weighted'),
        'F1-score': f1_score(y_val, y_pred, average='weighted'),
        'ROC AUC': roc_auc_score(y_val, y_proba, average='weighted', multi_class='ovo')
    }

results_all_genes = pd.DataFrame(results).T

# %%
print("Results using all genes:")
results_all_genes.sort_values('F1-score', ascending=False)


# %%
output_dir = Path('results/classifiers_evaluation')
output_dir.mkdir(parents=True, exist_ok=True)
results_all_genes.sort_values('F1-score', ascending=False).to_csv(output_dir / 'all_genes.csv')

# %% md
# ## Degs subset

# %%
X = degs_subset.T
y = metadata['PAM50']

# %%
# split the data into train test and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# %%
classifiers_degs = {
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=10000, random_state=42)),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'Neural Network': make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(200, 200, 100),
        max_iter=1000,
        early_stopping=True,
        random_state=42))
}

# %%
# train and evaluate classifiers
results = {}
for name, clf in (pbar := tqdm(classifiers_degs.items())):
    pbar.set_description(f'fitting {name}')
    clf.fit(X_train, y_train)
    pbar.set_description(f'predicting with {name}')
    y_pred = clf.predict(X_val)
    pbar.set_description(f'evaluating {name}')
    y_proba = (clf.predict_proba(X_val)
               if hasattr(clf, "predict_proba") else clf.decision_function(X_val))

    results[name] = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_val, y_pred, average='weighted'),
        'F1-score': f1_score(y_val, y_pred, average='weighted'),
        'ROC AUC': roc_auc_score(y_val, y_proba, average='weighted', multi_class='ovo')
    }


results_degs = pd.DataFrame(results).T

# %%
print("Results using DEGs:")
results_degs.sort_values('F1-score', ascending=False)


# %%
results_degs.sort_values('F1-score', ascending=False).to_csv(output_dir / 'degs.csv')


# %% md
# ML algorithms gave almost the same evaluation scores on all genes and on DEGs only,
# while training only on DEGs is significantly faster. Now let's tune the hyperparameters of
# top 3 algorithms (by F1-score): Gradient Boosting, Random Forest, SVM.
#
# Choice of F1-score over ROC AUC is determined by importance of good precision and recall,
# while diagnosing breast cancer subtype (high cost of false positives and false negatives),
# and because of imbalance in classes.


# %% md
# ## Tuning hyperparameters


# %% md
# ### Gradient Boosting

# %%
param_space_gb = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_leaf_nodes': [21, 31, 41]
}

grid_search_gb = GridSearchCV(
    classifiers_degs['Gradient Boosting'],
    param_space_gb,
    n_jobs=-1,
    scoring='f1_weighted',
    verbose=1,
)

grid_search_gb.fit(X_train, y_train)

print("Best parameters found for Gradient Boosting: ", grid_search_gb.best_params_)
print("Best F1-score for Gradient Boosting: ", grid_search_gb.best_score_)


# %% md
# Output:
#
# `{'learning_rate': 0.1, 'max_iter': 200, 'max_leaf_nodes': 31}`
#
# `0.8012793592379689`

# %% md
# ### SVM

# %%
param_space_svm = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto'],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid_search_svm = GridSearchCV(
    classifiers_degs['SVM'],
    param_space_svm,
    n_jobs=-1,
    scoring='f1_weighted',
    verbose=1,
    cv=5,
)

grid_search_svm.fit(X_train, y_train)

# %%
print("Best parameters found for SVM: ", grid_search_svm.best_params_)
print("Best F1-score for SVM: ", grid_search_svm.best_score_)


# %% md
# Output:
#
# `{'svc__C': 1, 'svc__gamma': 'scale', 'svc__kernel': 'linear'}`
#
# `0.8072037282506166`


# %% md
# ### Random Forest

# %%
param_space_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30]
}

grid_search_rf = GridSearchCV(
    classifiers_degs['Random Forest'],
    param_space_rf,
    n_jobs=-1,
    scoring='f1_weighted',
    verbose=1,
)

grid_search_rf.fit(X_train, y_train)


# %%
print("Best parameters found for Random Forest: ", grid_search_rf.best_params_)
print("Best F1-score for Random Forest: ", grid_search_rf.best_score_)


# %% md
# Output:
#
# `{'max_depth': None, 'max_features': None, 'n_estimators': 300}`
#
# `0.8172762597861276`, `0.814` for `n_estimators=100`


# %% md
# With both metrics (f1-weighted and f1-macro) Random Forest outperforms SVM and Gradient Boosting.
#
# Let's check, does n_estimators=100 vs 300 gives statistical difference in RF,
# and compare it with GB and SVM (with optimized parameters).

# %%
classifiers = {
    'rf1': RandomForestClassifier(
        **{'max_depth': None, 'max_features': None,
            'n_estimators': 100}),
    'rf2': RandomForestClassifier(
        **{'max_depth': None, 'max_features': None,
            'n_estimators': 300}),
    'gb': HistGradientBoostingClassifier(
        **{'learning_rate': 0.1, 'max_iter': 200,
            'max_leaf_nodes': 31}),
    'svm': make_pipeline(StandardScaler(), SVC(
        **{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}))
}

# %%
rkf = RepeatedKFold(n_splits=5, n_repeats=5,
                    random_state=42)

scores = {}
for name, clf in (pbar := tqdm(classifiers.items())):
    pbar.set_description(name)
    scores[name] = cross_val_score(
        clf, X_train, y_train, cv=rkf,
        n_jobs=-1, verbose=2,
        scoring='f1_weighted')


# %%
df_melted = pd.DataFrame(scores).melt(var_name='Classifier', value_name='F1-score')
df_melted.to_csv('results/classifiers_evaluation/tuning_results.csv', index=False)

# %%
scores_np = np.vstack((scores['rf1'], scores['rf2'], scores['gb'], scores['svm'])).T

friedman_stat, friedman_p = friedmanchisquare(
    scores['rf1'], scores['rf2'], scores['gb'], scores['svm'])
print(f'Friedman test statistic={friedman_stat}, p-value={friedman_p}')

# %%
if friedman_p < 0.05:
    nemenyi_result = sp.posthoc_nemenyi_friedman(scores_np)
    print("Nemenyi test results:\n", nemenyi_result)
else:
    print("No significant differences found among models using Friedman test.")

# %%
print('rf1: ', scores['rf1'].mean(), scores['rf1'].std())
print('rf2: ', scores['rf2'].mean(), scores['rf2'].std())
print('gb: ', scores['gb'].mean(), scores['gb'].std())
print('svm: ', scores['svm'].mean(), scores['svm'].std())


# %%
df_melted = pd.read_csv('results/classifiers_evaluation/tuning_results.csv')


fig = px.violin(df_melted, x='Classifier', y='F1-score',
                color='Classifier',
                box=True, points='all')
fig.update_layout(showlegend=False)

fig.add_trace(go.Scatter(
    x=['rf1', 'gb'],
    y=[0.92, 0.92],
    mode="lines",
    line=dict(color="black", width=1),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=['rf1', 'rf1'],
    y=[0.92, 0.91],
    mode="lines",
    line=dict(color="black", width=1),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=['gb', 'gb'],
    y=[0.92, 0.91],
    mode="lines",
    line=dict(color="black", width=1),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=['rf2', 'rf2'],
    y=[0.92, 0.93],
    mode="lines",
    line=dict(color="black", width=1),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=['rf2', 'svm'],
    y=[0.93, 0.93],
    mode="lines",
    line=dict(color="black", width=1),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=['svm', 'svm'],
    y=[0.93, 0.91],
    mode="lines",
    line=dict(color="black", width=1),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=['gb'],
    y=[0.93],
    mode="text",
    text=["*"],
    textposition="top center",
    textfont=dict(size=20),
    showlegend=False
))

fig.show()
save_fig(fig, 'results/figures/classifiers_comparison.html')

# %% md
# Results:
#
# mean and std:
#
# `rf1:  0.814729169017811 0.024106169329451487
# rf2:  0.8167264811361582 0.024605705361264024
# gb:  0.812763429534384 0.02430571116017395
# svm:  0.7928911647913511 0.030927744325978224`
#
# Nemenyi test results:
#
# `           0         1         2         3
# 0  1.000000  0.608787  0.900000  0.049306
# 1  0.608787  1.000000  0.639529  0.001000
# 2  0.900000  0.639529  1.000000  0.042567
# 3  0.049306  0.001000  0.042567  1.000000`
#
# So, rf1, rf2, and gb don't have significant differences,
# and they all perform better than SVM.
#
# Therefore, we'll choose rf1, as this model is less computationally expensive
# than rf2 and gb.
#


# %%
rf1 = classifiers['rf1']

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

# %%

rf1.fit(X_train_final, y_train_final)

# %%
y_test_pred = rf1.predict(X_test)
y_test_proba = rf1.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_roc_auc = roc_auc_score(y_test, y_test_proba, average='weighted', multi_class='ovo')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")

# %%
print(classification_report(y_test, y_test_pred))

# %% md
# ## Feature importance evaluation

# %% md
# ### Based on mean decrease in impurity


# %%
# feature importances for all dataset
importance_df_all = pd.DataFrame({
    'gene': X_train_final.columns,
    'All_mean': rf1.feature_importances_,
    'All_std': np.std([rf1.feature_importances_ for tree in rf1.estimators_], axis=0),
}
).set_index('gene')


# %%
# by subtypes
rf = OneVsRestClassifier(RandomForestClassifier(
    **{'max_depth': None, 'max_features': None, 'n_estimators': 100}))
rf.fit(X_train_final, y_train_final)

# %%
classes = rf.classes_

importance_dict = {}
for idx, class_label in enumerate(classes):
    class_importance = rf.estimators_[idx].feature_importances_
    importance_dict[f'{class_label}_mean'] = class_importance
    importance_dict[f'{class_label}_std'] = np.std([
        rf.estimators_[idx].feature_importances_ for tree
        in rf.estimators_[idx].estimators_])

importance_df = pd.DataFrame(importance_dict, index=X_train_final.columns)

# %%
importance_df = importance_df.join(importance_df_all)


# %%
output_dir = Path('results/feature_importance')
output_dir.mkdir(exist_ok=True, parents=True)
importance_df.to_csv(output_dir / 'feature_importance_rf_impurity.csv')

# %% md
# Let's get most important genes for each subtype


# %%
importance_df = pd.read_csv('results/feature_importance/feature_importance_rf_impurity.csv',
                            index_col=0)

# %%
important_genes = pd.DataFrame(
    {subtype.split('_', 1)[0]:
     importance_df.sort_values(
        by=subtype, ascending=False).index.tolist()
     for subtype in importance_df.columns if not subtype.endswith('_std')})
important_genes = important_genes[['All', 'LumA', 'LumB', 'Her2', 'Basal', 'Normal']]
important_genes.to_csv('results/feature_importance/important_genes_rf_impurity.csv', index=False)

# %%
important_genes.head(10)


# %% md
# ### Based on feature permutation

# %% md
# Let's remove multicollinear features and use only central genes from clusters,
# as multicollinear features might hide some important feature during permutation.

# %%
corr = spearmanr(X).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# %%
thresholds = np.linspace(0.1, 2.0, 20)
silhouette_scores = []

for t in thresholds:
    cluster_labels = hierarchy.fcluster(dist_linkage, t=t, criterion='distance')
    score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
    silhouette_scores.append(score)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=thresholds,
    y=silhouette_scores,
    mode='lines+markers',
    name='Silhouette Score'
))

fig.update_layout(
    xaxis_title='Threshold',
    yaxis_title='Silhouette Score'
)

fig.show()

# %%

optimal_threshold = thresholds[np.argmax(silhouette_scores)]
print('Optimal threshold:', optimal_threshold)

# %%
cluster_labels = hierarchy.fcluster(dist_linkage, optimal_threshold, criterion='distance')
genes_clusters = pd.DataFrame({'Gene': X.columns, 'Cluster': cluster_labels})


def select_representative_gene(cluster_genes):
    cluster_data = X[cluster_genes]
    mean_correlation = cluster_data.corr().mean().sort_values(ascending=False)
    return mean_correlation.index[0]


representative_genes = genes_clusters.groupby(
    'Cluster')['Gene'].apply(select_representative_gene).tolist()

X_train_sel = X_train_final[representative_genes]
X_test_sel = X_test[representative_genes]

# %%
clf_sel = RandomForestClassifier(
    **{'max_depth': None, 'max_features': None,
        'n_estimators': 100})
clf_sel.fit(X_train_sel, y_train_final)

# %%
y_test_pred_sel = clf_sel.predict(X_test_sel)
print(classification_report(y_test, y_test_pred_sel))


# %% md
# Perform feature permutation

# %%
# for all subtypes combined:
result = permutation_importance(clf_sel, X_train_sel, y_train_final,
                                scoring='f1_weighted', random_state=42, n_jobs=-1)

# %%
feature_importances_all = pd.DataFrame({
    'gene': X_train_sel.columns,
    'All_mean': result.importances_mean,
    'All_std': result.importances_std
}).set_index('gene').sort_values(by='All_mean', ascending=False)

# %%
# for each subtype:
y_pred = clf_sel.predict(X_train_sel)

feature_importances = {}

for subtype in (pbar := tqdm(y.unique())):
    pbar.set_description(subtype)
    if y_train_final[y_pred == subtype].shape != (0,):
        result = permutation_importance(
            clf_sel,
            X_train_sel[y_pred == subtype], y_pred[y_pred == subtype],
            scoring='f1_weighted', random_state=42, n_jobs=-1)
        feature_importances[subtype] = pd.DataFrame({
            'gene': X_train_sel.columns,
            f'{subtype}_mean': result.importances_mean,
            f'{subtype}_std': result.importances_std
        }).set_index('gene').sort_values(by=f'{subtype}_mean', ascending=False)

# %%
importance_permutation = pd.concat(list(feature_importances.values()) + [feature_importances_all],
                                   axis=1)

# %%
importance_permutation.to_csv(output_dir / 'feature_importance_rf_permutation.csv')

# %%
important_genes_permutation = pd.DataFrame({subtype.split('_', 1)[0]:
                                            importance_permutation.sort_values(
    by=subtype, ascending=False).index.tolist()
    for subtype in importance_permutation.columns if not subtype.endswith('_std')})


# %%
important_genes_permutation = important_genes_permutation[[
    'All', 'LumA', 'LumB', 'Her2', 'Basal', 'Normal']]
important_genes_permutation.to_csv(
    'results/feature_importance/important_genes_rf_permutation.csv', index=False)


# %%
important_genes_permutation.head(10)

# %%
important_genes_impurity = pd.read_csv(
    'results/feature_importance/important_genes_rf_impurity.csv')
important_genes_impurity.head(10)


# %% md
# Perform feature permutation without removing correlated features

# %%
# for all subtypes combined:
result = permutation_importance(rf1, X_train_final, y_train_final,
                                scoring='f1_weighted', random_state=42, n_jobs=-1)

# %%
feature_importances_all = pd.DataFrame({
    'gene': X_train_final.columns,
    'All_mean': result.importances_mean,
    'All_std': result.importances_std
}).set_index('gene').sort_values(by='All_mean', ascending=False)

# %%
# for each subtype:
y_pred = rf1.predict(X_train_final)

feature_importances = {}

for subtype in (pbar := tqdm(y.unique())):
    pbar.set_description(subtype)
    if y_train_final[y_pred == subtype].shape != (0,):
        result = permutation_importance(
            rf1,
            X_train_final[y_pred == subtype], y_pred[y_pred == subtype],
            scoring='f1_weighted', random_state=42, n_jobs=-1)
        feature_importances[subtype] = pd.DataFrame({
            'gene': X_train_final.columns,
            f'{subtype}_mean': result.importances_mean,
            f'{subtype}_std': result.importances_std
        }).set_index('gene').sort_values(by=f'{subtype}_mean', ascending=False)

# %%
importance_permutation = pd.concat(list(feature_importances.values()) + [feature_importances_all],
                                   axis=1)

# %%
output_dir = Path('results/feature_importance/')
importance_permutation.to_csv(output_dir / 'feature_importance_rf_permutation_all.csv')

# %%
important_genes_permutation = pd.DataFrame({subtype.split('_', 1)[0]:
                                            importance_permutation.sort_values(
    by=subtype, ascending=False).index.tolist()
    for subtype in importance_permutation.columns if not subtype.endswith('_std')})


# %%
important_genes_permutation = important_genes_permutation[[
    'All', 'LumA', 'LumB', 'Her2', 'Basal', 'Normal']]
important_genes_permutation.to_csv(
    'results/feature_importance/important_genes_rf_permutation_all.csv', index=False)


# %%
important_genes_permutation.head(10)


# %% md
# ### SHAP

# %%
class_names = rf1.classes_

explainer = shap.TreeExplainer(rf1)
shap_values = explainer.shap_values(X_train_final)

importance_shap_all = pd.DataFrame({
    'gene': X_train_final.columns,
    'All_mean': np.abs(shap_values).mean(axis=(0, 2)),
    'All_std': np.abs(shap_values).std(axis=(0, 2))
}).set_index('gene').sort_values(by='All_mean', ascending=False)

feature_importances = {}
for i, class_name in enumerate(class_names):
    feature_importances[class_name] = pd.DataFrame({
        'gene': X_train_final.columns,
        f'{class_name}_mean': np.abs(shap_values[:, :, i]).mean(axis=0),
        f'{class_name}_std': np.abs(shap_values[:, :, i]).std(axis=0)
    }).set_index('gene').sort_values(by=f'{class_name}_mean', ascending=False)

importance_shap = pd.concat(list(feature_importances.values()) + [importance_shap_all], axis=1)

# %%
importance_shap.to_csv(output_dir / 'feature_importance_rf_shap.csv')

# %%
important_genes_shap = pd.DataFrame({subtype.split('_', 1)[0]:
                                     importance_shap.sort_values(
    by=subtype, ascending=False).index.tolist()
    for subtype in importance_shap.columns if not subtype.endswith('_std')})


# %%
important_genes_shap = important_genes_shap[[
    'All', 'LumA', 'LumB', 'Her2', 'Basal', 'Normal']]
important_genes_shap.to_csv(
    'results/feature_importance/important_genes_rf_shap.csv', index=False)


# %%
important_genes_shap.head(10)


# %%
# plot most important features from all methods
method_to_df = {
    'Impurity': pd.read_csv(
        'results/feature_importance/feature_importance_rf_impurity.csv', index_col=0),
    'Permutation': pd.read_csv(
        'results/feature_importance/feature_importance_rf_permutation.csv', index_col=0),
    'SHAP': pd.read_csv(
        'results/feature_importance/feature_importance_rf_shap.csv', index_col=0)
}

method_to_axis_title = {
    'Impurity': 'Mean decrease in impurity',
    'Permutation': 'Mean F1-score decrease',
    'SHAP': 'SHAP value'
}

subtype_to_common_genes = {
    'All': [
        'MLPH',
        'FOXA1',
        'ESR1',
        'KRT14',
        'TPX2',
        'KRT5',
        'SGOL1',
        'LINC00504',
        'NEIL3',
    ],
    'LumA': [
        'TPX2',
        'KRT14',
        'SGOL1',
        'KRT5',
        'ESR1',
        'CENPA',
    ],
    'LumB': [
        'ESR1',
        'TPX2',
        'KRT14',
        'MLPH',
        'SGOL1',
        'KRT5',
        'NEIL3',
    ],
    'Her2': ['ESR1'],
    'Basal': [
        'MLPH',
        'FOXA1',
        'ESR1',
        'TTC6',
        'HJURP',
    ],
    'Normal': [
        'TPX2',
        'ESR1',
        'KIF20A',
        'KRT14',
    ]
}

for c, (method, imp_df) in enumerate(method_to_df.items()):
    for subtype, common_genes in subtype_to_common_genes.items():

        plot_df = imp_df.sort_values(by=f'{subtype}_mean', ascending=False).head(10)[
            [f'{subtype}_mean', f'{subtype}_std']]

        # calculate standard error instead of std:
        if method == 'Impurity':
            plot_df[f'{subtype}_std'] = plot_df[f'{subtype}_std'] / np.sqrt(100)  # n_estimators
        elif method == 'Permutation':
            plot_df[f'{subtype}_std'] = plot_df[f'{subtype}_std'] / np.sqrt(5)  # n_repeats
        else:
            if subtype == 'All':
                plot_df[f'{subtype}_std'] = (plot_df[f'{subtype}_std']  # n subtypes * n samples
                                             / np.sqrt(5 * X_train_final.shape[0]))
            else:
                plot_df[f'{subtype}_std'] = (plot_df[f'{subtype}_std']  # n samples
                                             / np.sqrt(X_train_final.shape[0]))

        fig = px.bar(
            plot_df.reset_index(),
            x='gene',
            y=f'{subtype}_mean',
            error_y=f'{subtype}_std',
            title=method,
            labels={'gene': '', f'{subtype}_mean': method_to_axis_title[method]},
            color='gene',
            color_discrete_map={gene: px.colors.qualitative.D3[c] for gene in plot_df.index},
            text=plot_df.index.map(lambda x: f'<b>{x}</b>' if x in common_genes else x)
        )
        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 1.1 * (plot_df[f'{subtype}_mean'].max()
                                   + plot_df[f'{subtype}_std'].max())])
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_xaxes(showticklabels=False, ticks='')
        save_fig(fig, f'results/figures/{method}_{subtype}.html'.lower())
        fig.show()
