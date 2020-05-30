from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestCentroid
import pandas as pd
import numpy as np
import copy

class DFAgglomerative(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name, columns=None, random_state=None,
                 eval_silhouette=False, eval_chi=False, eval_dbi=False,
                 **kwargs):
        self.cluster_name    = cluster_name
        self.columns         = columns
        self.random_state    = random_state
        self.model           = AgglomerativeClustering(**kwargs)
        self.eval_silhouette = eval_silhouette
        self.eval_chi        = eval_chi
        self.eval_dbi        = eval_dbi
        self.transform_cols  = None
        self.eval_df         = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        self.centroid_df    = pd.DataFrame(
            self.__calc_centroids(
                X[self.transform_cols],
                self.model.fit_predict(X[self.transform_cols])
            ),
            columns=self.transform_cols
        )
        self.centroid_df['Cluster'] = [f'Cluster {x}' for x in np.unique(self.model.labels_)]
        self.centroid_df.set_index('Cluster', inplace=True)
        self.centroid_df.index.name = None

        # Evaluation
        self.eval_df = pd.DataFrame({
            'n_cluster': [x+1 for x in range(self.model.n_clusters)],
        })

        if any([self.eval_silhouette, self.eval_chi, self.eval_dbi]):
            silhouettes = []
            chis        = []
            dbis        = []

            self.eval_df['centroid'] = self.eval_df['n_cluster'].apply(lambda x: [])

            tmp_X = X[self.transform_cols].copy()
            for x in range(self.model.n_clusters):
                model = copy.deepcopy(self.model)
                model.n_clusters = x+1
                model.fit(tmp_X)

                # Cluster centroid
                self.eval_df.at[x, 'centroid'] = self.__calc_centroids(tmp_X, model.fit_predict(tmp_X))

                # Reference: https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
                if self.eval_silhouette:
                    silhouettes.append(np.nan if x == 0 else silhouette_score(tmp_X, model.labels_, metric='euclidean', random_state=self.random_state))

                # Reference: https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski-harabasz-ch-criterion
                if self.eval_chi:
                    chis.append(np.nan if x == 0 else calinski_harabasz_score(tmp_X, model.labels_))

                # Reference: https://stackoverflow.com/questions/59279056/davies-bouldin-index-higher-or-lower-score-better
                if self.eval_dbi:
                    dbis.append(np.nan if x == 0 else davies_bouldin_score(tmp_X, model.labels_))

            if self.eval_silhouette:
                self.eval_df['silhouette'] = silhouettes

            if self.eval_chi:
                self.eval_df['calinski_harabasz'] = chis

            if self.eval_dbi:
                self.eval_df['davies_bouldin'] = dbis

        return self
    
    def __calc_centroids(self, X, y):
        if len(np.unique(y)) <= 1:
            return []

        # Reference: https://stackoverflow.com/questions/56456572/how-to-get-agglomerative-clustering-centroid-in-python-scikit-learn
        return NearestCentroid().fit(X, y).centroids_

    # NOTE: AgglomerativeClustering does not have predict()
    def __predict(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.cluster_name] = self.model.fit_predict(X[self.transform_cols])
        new_X[self.cluster_name] = 'Cluster ' + new_X[self.cluster_name].astype(str)

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).__predict(X)