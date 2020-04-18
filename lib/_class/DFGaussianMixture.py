from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import copy

class DFGaussianMixture(BaseEstimator, ClusterMixin):
    def __init__(self, cluster_name, eval_aic=False, eval_bic=False, eval_silhouette=False, columns=None, **kwargs):
        self.cluster_name    = cluster_name
        self.eval_aic        = eval_aic
        self.eval_bic        = eval_bic
        self.eval_silhouette = eval_silhouette
        self.columns         = columns
        self.model           = GaussianMixture(**kwargs)
        self.transform_cols  = None
        self.eval_df         = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        self.eval_df = pd.DataFrame({
            'n_cluster': [x+1 for x in range(self.model.n_components)]
        })

        if self.eval_aic or self.eval_bic or self.eval_silhouette:
            aics        = []
            bics        = []
            silhouettes = []
            self.eval_df['centroid']  = self.eval_df['n_cluster'].apply(lambda x: [])
            self.eval_df['converged'] = [None for _ in range(self.model.n_components)]

            tmp_X = X[self.transform_cols].copy()
            for x in range(self.model.n_components):
                model = copy.deepcopy(self.model)
                model.n_components = x+1
                model.fit(tmp_X)

                self.eval_df.at[x, 'converged'] = model.converged_

                # Cluster centroid
                # Reference: https://stackoverflow.com/questions/47412749/how-can-i-get-a-representative-point-of-a-gmm-cluster
                centroids = np.empty(shape=(model.n_components, tmp_X.shape[1]))
                for x in range(model.n_components):
                    density         = multivariate_normal(mean=model.means_[x], cov=model.covariances_[x]).logpdf(tmp_X)
                    centroids[x, :] = tmp_X.loc[np.argmax(density)].values
                self.eval_df.at[x, 'centroid'] = centroids

                # Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
                if self.eval_aic:
                    aics.append(model.aic(tmp_X))

                if self.eval_bic:
                    bics.append(model.bic(tmp_X))

                # Reference: https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6
                if self.eval_silhouette:
                    silhouettes.append(np.nan if x == 0 else silhouette_score(tmp_X, model.predict(tmp_X), metric='euclidean', random_state=model.random_state))

            if self.eval_aic:
                self.eval_df['aic'] = aics

            if self.eval_bic:
                self.eval_df['bic'] = bics

            if self.eval_silhouette:
                self.eval_df['silhouette'] = silhouettes

        return self
    
    def predict(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.cluster_name] = self.model.predict(X[self.transform_cols])

        return new_X
    
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)