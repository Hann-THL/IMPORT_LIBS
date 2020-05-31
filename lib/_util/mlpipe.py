# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score

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
def eval_classif(y_true, y_pred, multi_class='raise'):
    cofmat_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cofmat_df.index.name   = 'True'
    cofmat_df.columns.name = 'Pred'

    roc_auc = roc_auc_score(
        pd.get_dummies(y_true) if multi_class in ['ovr', 'ovo'] else y_true,
        pd.get_dummies(y_pred) if multi_class in ['ovr', 'ovo'] else y_pred,
        multi_class=multi_class
    )
    kappa   = cohen_kappa_score(y_true, y_pred)

    print(cofmat_df)
    print()
    print(classification_report(y_true, y_pred, digits=5))
    print(f'ROC-AUC: {roc_auc : .5f}')
    print(f'Kappa:   {kappa :.5f}')