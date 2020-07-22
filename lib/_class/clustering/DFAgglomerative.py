from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

class DFAgglomerative(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name='Agglo', columns=None, random_state=None,
                 clusters=None, eval_silhouette=False, eval_chi=False, eval_dbi=False, eval_sample_size=None,
                 **kwargs):
        if any([eval_silhouette, eval_chi, eval_dbi]):
            assert clusters is not None, 'clusters should consists of [n_cluster] for Agglomerative evaluation.'

        self.cluster_name     = cluster_name
        self.columns          = columns
        self.random_state     = random_state
        self.model            = AgglomerativeClustering(**kwargs)
        self.clusters         = clusters
        self.eval_silhouette  = eval_silhouette
        self.eval_chi         = eval_chi
        self.eval_dbi         = eval_dbi
        self.eval_sample_size = eval_sample_size
        self.transform_cols   = None
        self.eval_df          = None
        self.centroid_df      = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Evaluation
        if any([self.eval_silhouette, self.eval_chi, self.eval_dbi]):
            silhouettes = []
            chis        = []
            dbis        = []

            self.eval_df = pd.DataFrame({
                'n_cluster': self.clusters,
            })
            self.eval_df['centroid'] = self.eval_df['n_cluster'].apply(lambda x: [])

            tmp_X = X[self.transform_cols].copy()
            index = 0
            for n_cluster in tqdm(self.eval_df['n_cluster'].values):
                model = copy.deepcopy(self.model)
                model.n_clusters = n_cluster
                model.fit(tmp_X)

                # Cluster centroid
                self.eval_df.at[index, 'centroid'] = self.__calc_centroids(tmp_X, model.labels_)

                # Reference: https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
                if self.eval_silhouette:
                    silhouettes.append(np.nan if n_cluster <= 1 else silhouette_score(tmp_X, model.labels_, sample_size=self.eval_sample_size, metric='euclidean', random_state=self.random_state))

                # Reference: https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski-harabasz-ch-criterion
                if self.eval_chi:
                    chis.append(np.nan if n_cluster <= 1 else calinski_harabasz_score(tmp_X, model.labels_))

                # Reference: https://stackoverflow.com/questions/59279056/davies-bouldin-index-higher-or-lower-score-better
                if self.eval_dbi:
                    dbis.append(np.nan if n_cluster <= 1 else davies_bouldin_score(tmp_X, model.labels_))

                index += 1

            if self.eval_silhouette:
                self.eval_df['silhouette'] = silhouettes

            if self.eval_chi:
                self.eval_df['calinski_harabasz'] = chis

            if self.eval_dbi:
                self.eval_df['davies_bouldin'] = dbis

        # Train
        else:
            self.model.fit(X[self.transform_cols])

            self.centroid_df = pd.DataFrame(
                self.__calc_centroids(
                    X[self.transform_cols],
                    self.model.labels_
                ),
                columns=self.transform_cols
            )
            self.centroid_df['Cluster'] = [f'Cluster {x}' for x in np.unique(self.model.labels_)]
            self.centroid_df.set_index('Cluster', inplace=True)
            self.centroid_df.index.name = None

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
        new_X[self.cluster_name] = self.model.labels_
        new_X[self.cluster_name] = 'Cluster ' + new_X[self.cluster_name].astype(str)

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).__predict(X)