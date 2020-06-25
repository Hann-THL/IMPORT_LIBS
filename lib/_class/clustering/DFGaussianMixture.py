from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import multivariate_normal
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

class DFGaussianMixture(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name='GMM', columns=None,
                 eval_aic=False, eval_bic=False, eval_silhouette=False, eval_chi=False, eval_dbi=False, eval_sample_size=None,
                 **kwargs):
        self.cluster_name     = cluster_name
        self.columns          = columns
        self.model            = GaussianMixture(**kwargs)
        self.eval_aic         = eval_aic
        self.eval_bic         = eval_bic
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
        if any([self.eval_aic, self.eval_bic, self.eval_silhouette, self.eval_chi, self.eval_dbi]):
            aics        = []
            bics        = []
            silhouettes = []
            chis        = []
            dbis        = []

            self.eval_df = pd.DataFrame({
                'n_cluster': [x+1 for x in range(self.model.n_components)]
            })
            self.eval_df['centroid']  = self.eval_df['n_cluster'].apply(lambda x: [])
            self.eval_df['converged'] = [None for _ in range(self.model.n_components)]

            tmp_X = X[self.transform_cols].copy()
            for x in tqdm(range(self.model.n_components)):
                model = copy.deepcopy(self.model)
                model.n_components = x+1
                model.fit(tmp_X)

                self.eval_df.at[x, 'converged'] = model.converged_

                # Cluster centroid
                self.eval_df.at[x, 'centroid'] = self.__calc_centroids(model, tmp_X)

                # Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
                if self.eval_aic:
                    aics.append(model.aic(tmp_X))

                if self.eval_bic:
                    bics.append(model.bic(tmp_X))

                # Reference: https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
                if self.eval_silhouette:
                    silhouettes.append(np.nan if x == 0 else silhouette_score(tmp_X, model.predict(tmp_X), sample_size=self.eval_sample_size, metric='euclidean', random_state=model.random_state))

                # Reference: https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski-harabasz-ch-criterion
                if self.eval_chi:
                    chis.append(np.nan if x == 0 else calinski_harabasz_score(tmp_X, model.predict(tmp_X)))

                # Reference: https://stackoverflow.com/questions/59279056/davies-bouldin-index-higher-or-lower-score-better
                if self.eval_dbi:
                    dbis.append(np.nan if x == 0 else davies_bouldin_score(tmp_X, model.predict(tmp_X)))

            if self.eval_aic:
                self.eval_df['akaike'] = aics

            if self.eval_bic:
                self.eval_df['bayesian'] = bics

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
                self.__calc_centroids(self.model, X[self.transform_cols]),
                columns=self.transform_cols
            )
            self.centroid_df['Cluster'] = [f'Cluster {x}' for x in self.centroid_df.index]
            self.centroid_df.set_index('Cluster', inplace=True)
            self.centroid_df.index.name = None

        return self
    
    def __calc_centroids(self, model, X):
        # Reference: https://stackoverflow.com/questions/47412749/how-can-i-get-a-representative-point-of-a-gmm-cluster
        centroids = np.empty(shape=(model.n_components, X.shape[1]))
        for n in range(model.n_components):
            density         = multivariate_normal(mean=model.means_[n], cov=model.covariances_[n]).logpdf(X)
            centroids[n, :] = X.loc[np.argmax(density)].values

        return centroids

    def predict(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.cluster_name] = self.model.predict(X[self.transform_cols])
        new_X[self.cluster_name] = 'Cluster ' + new_X[self.cluster_name].astype(str)

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def predict_proba(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = pd.concat([
            X,
            pd.DataFrame(
                self.model.predict_proba(X[self.transform_cols]),
                columns=[f'{self.cluster_name} Cluster {x}' for x in range(len(self.centroid_df))]
            )
        ], axis=1)

        return new_X