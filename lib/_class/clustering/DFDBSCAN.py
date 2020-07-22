from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

# Reference: https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea
class DFDBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name='DBSCAN', columns=None, random_state=None,
                 eps_samples_tuples=None, eval_cluster=False, eval_silhouette=False, eval_chi=False, eval_dbi=False, eval_sample_size=None,
                 **kwargs):
        if any([eval_cluster, eval_silhouette, eval_chi, eval_dbi]):
            assert eps_samples_tuples is not None, 'eps_samples_tuples should consists of [(eps, min_samples)] for DBSCAN evaluation.'

        self.cluster_name       = cluster_name
        self.columns            = columns
        self.random_state       = random_state
        self.model              = DBSCAN(**kwargs)
        self.eps_samples_tuples = eps_samples_tuples
        self.eval_cluster       = eval_cluster
        self.eval_silhouette    = eval_silhouette
        self.eval_chi           = eval_chi
        self.eval_dbi           = eval_dbi
        self.eval_sample_size   = eval_sample_size
        self.transform_cols     = None
        self.eval_df            = None
        self.centroid_df        = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Evaluation
        if any([self.eval_cluster, self.eval_silhouette, self.eval_chi, self.eval_dbi]):
            n_clusters   = []
            n_noises     = []
            silhouettes1 = []
            silhouettes2 = []
            chis1        = []
            chis2        = []
            dbis1        = []
            dbis2        = []

            self.eval_df                = pd.DataFrame()
            self.eval_df['eps']         = [x[0] for x in self.eps_samples_tuples]
            self.eval_df['min_samples'] = [x[1] for x in self.eps_samples_tuples]
            self.eval_df['centroid']    = self.eval_df['eps'].apply(lambda x: [])

            tmp_X = X[self.transform_cols].copy()
            index = 0
            for eps, min_samples in tqdm(self.eps_samples_tuples):
                model = copy.deepcopy(self.model)
                model.eps = eps
                model.min_samples = min_samples
                model.fit(tmp_X)

                # Cluster centroid
                # Exclude calculating centroid of noise cluster
                tmp_df = pd.concat([tmp_X, pd.Series(model.labels_, name='Cluster')], axis=1)
                tmp_df = tmp_df[tmp_df['Cluster'] != -1].reset_index(drop=True)
                self.eval_df.at[index, 'centroid'] = self.__calc_centroids(tmp_df[self.transform_cols], tmp_df['Cluster'])

                tmp_X2  = tmp_X.copy()
                tmp_X2  = pd.concat([tmp_X2, pd.Series(model.labels_, name='Cluster')], axis=1)
                labels2 = tmp_X2[tmp_X2['Cluster'] != -1]['Cluster'].values
                tmp_X2  = tmp_X2[tmp_X2['Cluster'] != -1].drop(columns=['Cluster']).values

                # Reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
                n_cluster  = len(np.unique(model.labels_))
                n_cluster2 = len(np.unique(labels2))
                if self.eval_cluster:
                    n_clusters.append(n_cluster)
                    n_noises.append(np.sum(np.where(model.labels_ == -1, 1, 0)))

                # Reference: https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
                if self.eval_silhouette:
                    kwargs = {
                        'metric':       'euclidean',
                        'sample_size':  self.eval_sample_size,
                        'random_state': self.random_state
                    }
                    silhouettes1.append(np.nan if n_cluster <= 1 else silhouette_score(tmp_X, model.labels_, **kwargs))
                    silhouettes2.append(np.nan if n_cluster2 <= 1 else silhouette_score(tmp_X2, labels2, **kwargs))

                # Reference: https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski-harabasz-ch-criterion
                if self.eval_chi:
                    chis1.append(np.nan if n_cluster <= 1 else calinski_harabasz_score(tmp_X, model.labels_))
                    chis2.append(np.nan if n_cluster2 <= 1 else calinski_harabasz_score(tmp_X2, labels2))

                # Reference: https://stackoverflow.com/questions/59279056/davies-bouldin-index-higher-or-lower-score-better
                if self.eval_dbi:
                    dbis1.append(np.nan if n_cluster <= 1 else davies_bouldin_score(tmp_X, model.labels_))
                    dbis2.append(np.nan if n_cluster2 <= 1 else davies_bouldin_score(tmp_X2, labels2))

                index += 1

            if self.eval_cluster:
                self.eval_df['n_cluster'] = n_clusters
                self.eval_df['n_noise']   = n_noises

            if self.eval_silhouette:
                self.eval_df['silhouette']           = silhouettes1
                self.eval_df['silhouette_w/o_noise'] = silhouettes2

            if self.eval_chi:
                self.eval_df['calinski_harabasz']           = chis1
                self.eval_df['calinski_harabasz_w/o_noise'] = chis2

            if self.eval_dbi:
                self.eval_df['davies_bouldin']           = dbis1
                self.eval_df['davies_bouldin_w/o_noise'] = dbis2

        # Train
        else:
            self.model.fit(X[self.transform_cols])

            # Exclude calculating centroid of noise cluster
            tmp_df = pd.concat([X[self.transform_cols], pd.Series(self.model.labels_, name='Cluster')], axis=1)
            tmp_df = tmp_df[tmp_df['Cluster'] != -1].reset_index(drop=True)

            self.centroid_df = pd.DataFrame(
                self.__calc_centroids(
                    tmp_df[self.transform_cols],
                    tmp_df['Cluster']
                ),
                columns=self.transform_cols
            )
            self.centroid_df['Cluster'] = [f'Cluster {x}' for x in np.unique(self.model.labels_) if x != -1]
            self.centroid_df.set_index('Cluster', inplace=True)
            self.centroid_df.index.name = None

        return self
    
    def __calc_centroids(self, X, y):
        if len(np.unique(y)) <= 1:
            return []

        # Reference: https://stackoverflow.com/questions/56456572/how-to-get-agglomerative-clustering-centroid-in-python-scikit-learn
        return NearestCentroid().fit(X, y).centroids_

    # NOTE: DBSCAN does not have predict()
    def __predict(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.cluster_name] = self.model.labels_
        new_X[self.cluster_name] = 'Cluster ' + new_X[self.cluster_name].astype(str)

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).__predict(X)