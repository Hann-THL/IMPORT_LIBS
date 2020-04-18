from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import copy

# Reference: https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea
class DFDBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name, eps_samples_tuples=None, eval_cluster=False, eval_silhouette=False, columns=None, random_state=None, **kwargs):
        if eval_cluster or eval_silhouette:
            assert eps_samples_tuples is not None, 'eps_samples_tuples should consists of [(eps, min_samples)] for DBSCAN evaluation.'

        self.cluster_name       = cluster_name
        self.eps_samples_tuples = eps_samples_tuples
        self.eval_cluster       = eval_cluster
        self.eval_silhouette    = eval_silhouette
        self.columns            = columns
        self.random_state       = random_state
        self.model              = DBSCAN(**kwargs)
        self.transform_cols     = None
        self.eval_df            = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        self.eval_df = pd.DataFrame()

        if self.eval_cluster or self.eval_silhouette:
            n_clusters  = []
            n_noises    = []
            silhouettes = []

            self.eval_df['eps']         = [x[0] for x in self.eps_samples_tuples]
            self.eval_df['min_samples'] = [x[1] for x in self.eps_samples_tuples]

            tmp_X = X[self.transform_cols].copy()
            for index, (eps, min_samples) in enumerate(self.eps_samples_tuples):
                model = copy.deepcopy(self.model)
                model.eps = eps
                model.min_samples = min_samples
                model.fit(tmp_X)

                # Reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
                n_cluster = len(np.unique(model.labels_))
                if self.eval_cluster:
                    n_clusters.append(n_cluster)
                    n_noises.append(np.sum(np.where(model.labels_ == -1, 1, 0)))

                # Reference: https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
                if self.eval_silhouette:
                    silhouettes.append(np.nan if n_cluster == 1 else silhouette_score(tmp_X, model.labels_, metric='euclidean', random_state=self.random_state))

            if self.eval_cluster:
                self.eval_df['n_cluster'] = n_clusters
                self.eval_df['n_noise']   = n_noises

            if self.eval_silhouette:
                self.eval_df['silhouette'] = silhouettes

        return self
    
    # NOTE: DBCSAN does not have predict()
    def __predict(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.cluster_name] = self.model.fit_predict(X[self.transform_cols])

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).__predict(X)