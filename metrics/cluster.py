"""
Clustering: K-Means, Hierarchical Clustering, density estimation, KNN
To see whether the saliency maps have clusters

"""

## ------------------
## --- Third-Party ---
## ------------------
import os
import sys
sys.path.append('..')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)

import numpy as np
import torch as t
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering



class Kmean_sklearn:
    """
    K-Means alg solved using Sklearn
    """
    def __init__(self, n_clusters=10, random_state=42):
        self.estimator = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, samples):
        """
        Parameters
        ----------
        samples: the samples to be estimated

        Returns
        -------
        kmeans, which has attributes (cluster_centers_, labels_, inertia_, n_iter_)
        """
        kmeans = self.estimator.fit(samples)

        return kmeans


class Agglomerative_cluster_sklearn:
    """
    Agglomerative: a “bottom up” approach where elements start as individual clusters and
                   clusters are merged as one moves up the hierarchy
    """
    def __init__(self, n_clusters=10,
                 affinity="euclidean",
                 linkage="ward"):
        self.estimator = AgglomerativeClustering(n_clusters=n_clusters,
                                                 affinity=affinity,
                                                 linkage=linkage)
    def fit(self, samples):
        """

        Parameters
        ----------
        samples: the samples to be estimated

        Returns
        -------
        agglomerative cluster: has attributes (n_clusters_, labels_, n_leaves_, n_connected_components_,
                                               children_, distances_)
        """
        cluster = self.estimator.fit(samples)

        return cluster