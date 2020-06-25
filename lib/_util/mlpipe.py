# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np
from tqdm import tqdm



# PRE-PROCESS
def xy_split(df, target):
    X = df[[x for x in df.columns if x != target]].copy()
    y = df[target].copy()

    return X, y

# Reference:
# - https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
# - https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_memory_usage(df):
    def mb_memory(df):
        return df.memory_usage().sum() / 1024**2
    
    new_df = df.copy()
    original_memory = mb_memory(new_df)
    print(f'Original memory usage:  {original_memory :.2f} MB')

    for column in tqdm(new_df.select_dtypes(include=['number', 'object']).columns):
        dtype = new_df[column].dtype.name.lower()

        # Object fields
        if dtype == 'object':
            new_df[column] = new_df[column].astype('category')
            continue

        min_value = new_df[column].min()
        max_value = new_df[column].max()

        # Integer fields
        if dtype.startswith('int'):
            # Exclude fields with missing values
            if new_df[column].isna().sum() != 0:
                continue

            for dtype in [np.uint8, np.uint16, np.uint32, np.uint64,
                          np.int8, np.int16, np.int32, np.int64]:
                if min_value >= np.iinfo(dtype).min and max_value <= np.iinfo(dtype).max:
                    new_df[column] = new_df[column].astype(dtype)
                    break
        
        # Decimal fields
        else:
            for dtype in [np.float16, np.float32, np.float64]:
                if min_value >= np.finfo(dtype).min and max_value <= np.finfo(dtype).max:
                    new_df[column] = new_df[column].astype(dtype)
                    break

    optimize_memory = mb_memory(new_df)
    print(f'Optimized memory usage: {optimize_memory :.2f} MB')
    print(f'Memory decreased by {(original_memory - optimize_memory) / original_memory * 100:.2f} %')
    
    return new_df

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
    matrix_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
    matrix_df.index.name   = 'True'
    matrix_df.columns.name = 'Pred'

    kappa   = cohen_kappa_score(y_true, y_pred)
    # Reference: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    y_prob  = y_pred if y_prob is None else y_prob
    roc_auc = roc_auc_score(y_true, y_prob, multi_class=multi_class)

    pr_auc = np.NaN
    if y_true.nunique() == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc               = auc(recall, precision)

    if show_evaluation:
        print(matrix_df)
        print()
        print(classification_report(y_true, y_pred, digits=5))
        print(f'ROC-AUC: {roc_auc :.5f}')
        print(f'PR AUC:  {pr_auc :.5f}')
        print(f'Kappa:   {kappa :.5f}')

    if return_evaluation:
        return {
            'matrix':  matrix_df,
            'report':  classification_report(y_true, y_pred, digits=5, output_dict=True),
            'roc_auc': roc_auc,
            'pr_auc':  pr_auc,
            'kappa':   kappa
        }