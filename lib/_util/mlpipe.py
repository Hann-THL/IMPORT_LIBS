# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, cohen_kappa_score

import pandas as pd


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



# EVALUATION
def eval_classif(y_true, y_pred, y_prob=None, multi_class='raise', return_evaluation=False, show_evaluation=True):
    cofmat_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cofmat_df.index.name   = 'True'
    cofmat_df.columns.name = 'Pred'

    kappa   = cohen_kappa_score(y_true, y_pred)
    # Reference: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    y_prob  = y_pred if y_prob is None else y_prob
    roc_auc = roc_auc_score(y_true, y_prob, multi_class=multi_class)
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