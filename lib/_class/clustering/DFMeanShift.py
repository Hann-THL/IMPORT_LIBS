from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

class DFMeanShift(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name='MeanShift', columns=None, random_state=None,
                 bandwidths=None, eval_cluster=False, eval_silhouette=False, eval_chi=False, eval_dbi=False, eval_sample_size=None,
                 **kwargs):
        if any([eval_cluster, eval_silhouette, eval_chi, eval_dbi]):
            assert bandwidths is not None, 'bandwidths is required for MeanShift evaluation.'

        self.cluster_name       = cluster_name
        self.columns            = columns
        self.random_state       = random_state
        self.model              = MeanShift(**kwargs)
        self.bandwidths         = bandwidths
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

            self.eval_df              = pd.DataFrame()
            self.eval_df['bandwidth'] = self.bandwidths
            self.eval_df['centroid']  = self.eval_df['bandwidth'].apply(lambda x: [])

            tmp_X = X[self.transform_cols].copy()
            index = 0
            for bandwidth in tqdm(self.bandwidths):
                model = copy.deepcopy(self.model)
                model.bandwidth = bandwidth
                model.fit(tmp_X)

                # Cluster centroid
                self.eval_df.at[index, 'centroid'] = model.cluster_centers_

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

            self.centroid_df = pd.DataFrame(
                self.model.cluster_centers_,
                columns=self.transform_cols
            )
            self.centroid_df['Cluster'] = [f'Cluster {x}' for x in np.unique(self.model.labels_) if x != -1]
            self.centroid_df.set_index('Cluster', inplace=True)
            self.centroid_df.index.name = None

        return self
    
    def predict(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.cluster_name] = self.model.predict(X[self.transform_cols])
        new_X[self.cluster_name] = 'Cluster ' + new_X[self.cluster_name].astype(str)

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)