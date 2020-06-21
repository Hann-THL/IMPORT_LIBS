# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np



# PRE-PROCESS
def xy_split(df, target):
    X = df[[x for x in df.columns if x != target]].copy()
    y = df[target].copy()

    return X, y

def dataset_split(X, y, reset_index=True, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    if reset_index:
        X_train = X_train.reset_index(drop=True)
        X_test  = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def class_weight(y, normalize=False):
    classes     = np.unique(y)
    weights     = compute_class_weight('balanced', classes, y)
    weight_dict = {classes[i]: x for i,x in enumerate(weights)}
    
    if normalize:
        return {k: v / np.sum(list(weight_dict.values())) for k,v in weight_dict.items()}
    return weight_dict

# Reference:
# - https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/?fbclid=IwAR1PcEicqDXadG9hsNE-Tf4RQQ_DpIaCV4LRcuizGbTC9Ek5PiMbB_x26bU
# - https://www.youtube.com/watch?v=D6AChZlN5m0
def class_ratio(y, rounding=None, normalize=False):
    roundings = [None, 'round', 'ceil', 'floor']
    assert rounding in roundings, f'rounding not in valid list: {roundings}'
    
    count_series = y.value_counts().sort_index()
    weight_dict  = {k: v for k,v in count_series.items()}
    weight_dict  = {x: weight_dict[0] / weight_dict[i] for i,x in enumerate(count_series.keys())}
    
    if rounding == 'round':
        weight_dict = {k: int(np.round(v)) for k,v in weight_dict.items()}
    elif rounding == 'ceil':
        weight_dict = {k: int(np.ceil(v)) for k,v in weight_dict.items()}
    elif rounding == 'floor':
        weight_dict = {k: int(np.floor(v)) for k,v in weight_dict.items()}
    
    if normalize:
        return {k: v / np.sum(list(weight_dict.values())) for k,v in weight_dict.items()}
    return weight_dict



# EVALUATION
def eval_classif(y_true, y_pred, y_prob=None, multi_class='raise', return_evaluation=False, show_evaluation=True):
    is_binary = y_true.nunique() == 2

    cofmat_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cofmat_df.index.name   = 'True'
    cofmat_df.columns.name = 'Pred'

    kappa   = cohen_kappa_score(y_true, y_pred)
    # Reference: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    y_prob  = y_pred if y_prob is None else y_prob
    roc_auc = roc_auc_score(y_true, y_prob, multi_class=multi_class)

    pr_auc = np.NaN
    if is_binary:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc               = auc(recall, precision)

    if show_evaluation:
        print(cofmat_df)
        print()
        print(classification_report(y_true, y_pred, digits=5))
        print(f'ROC-AUC: {roc_auc :.5f}')
        print(f'PR AUC:  {pr_auc :.5f}')
        print(f'Kappa:   {kappa :.5f}')

    if return_evaluation:
        return {
            'report':  classification_report(y_true, y_pred, digits=5, output_dict=True),
            'roc_auc': roc_auc,
            'pr_auc':  pr_auc,
            'kappa':   kappa
        }