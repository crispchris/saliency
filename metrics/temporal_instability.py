"""
Temporal Instability:
    Inspired from Location Instability(filter's location instability), which was used in Image data
    The deviation of the distance between the inferred position p^ and specific ground truth landmark
The idea of temporal instability: Not on filter, but average per Visualization Method,
    Threshold: maybe 0.7 * Max Point
    Histogram through times series(Axis)
    Deviation of histogram per Class (through TestSet)
    Average per Visualization Method
"""
## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import torch as t
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
## -----------
## --- Own ---
## -----------
# from visualize_mechanism.cam import interpolate_smooth_ucr
from trainhelper.dataset import Dataset
## Device Configuration
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

## Datasets for UCR
class Temporal_instability:
    def __init__(self, testset: Dataset):
        self.testset = testset
        self.data_len = len(testset.labels)
        self.num_classes = len(np.unique(testset.labels))


    def histogram_topfeatures(self, batch_samples: np.ndarray,
                              batch_labels: np.ndarray,
                              batch_explanation: np.ndarray,
                              percent_of_top_features: float = 0.1):
        """
        According to given Explanation, return the histogram of top features
        (for a sample of explanation, the highest top points count as hits for histogram)
        Histograms distinguish between C (the dimension of sensors)

        Parameters
        ----------
        batch_samples (np.ndarray) : batch samples [B, C, L]
        batch_labels (List) :  batch labels
        batch_explanation (np.ndarray) : batch explanation from a vis. Method [B, C, L]
                                        The explanation should be (only the right labels, wrong predictions should
                                        be taken out)
        number_of_top_features (int) : the number of features in a sample explanation, will be counted as hits
                                        for example: 2, it will computed until the top two highlighted features
                                        in sample explanation
        Returns
        -------
        Histogram of features (np.ndarray): [Nc, C, L]
                                            Nc is number of classes,
                                            C the dimension of sensors(features)
                                            L the time
        """
        assert batch_samples.shape[0] == batch_labels.shape[0]
        assert batch_samples.shape == batch_explanation.shape

        histogram = np.zeros((self.num_classes, self.testset.data.shape[1], self.testset.data.shape[-1]))

        top_features = int(histogram.shape[-1] * percent_of_top_features)
        ## values, indexes
        feature_idxs = np.flip(np.argsort(batch_explanation, axis=-1), axis=-1)
        top_features = feature_idxs[:, :, :top_features]

        x = np.linspace(0, 1, top_features.shape[2]).astype("int32")
        y = np.linspace(0, top_features.shape[1] - 1, top_features.shape[1]).astype("int32")
        xv, yv = np.meshgrid(x, y)
        ## compute histogram per class
        for num, label in enumerate(batch_labels):
            histogram[label][yv, top_features[num]] += 1

        return histogram

    def temporal_topfeatures(self, histogram: np.ndarray,
                             number_of_featuresgroups: int):
        """
        According to the histogram, return the indexes of the top features

        Parameters
        ----------
        histogram (np.ndarray) : the histogram of feature maps
        number_of_featuresgroups (int): the number of feature groups

        Returns
        -------
        top_features
        The Indexes of the top features [Nc, C, number_of_featuresgroups]
                                        Nc : the number of classes
                                        C : the dimension of sensors (features)
                                        number_of_featuresgroups : the number of the most important feature groups
        """
        ## values, indexes
        feature_idxs = np.flip(np.argsort(histogram, axis=-1), axis=-1)
        top_features = feature_idxs[:, :, :number_of_featuresgroups]

        return top_features

    ## TODO: die Idee mit Threshold
    # def get_histogram(self, threshold: float = 0.7):
    #     histogram = np.zeros((self.num_classes, self.testset.data.shape[1], self.testset.data.shape[-1]))  ## UCR data shape = [B, dim, Length]
    #     for i in range(self.num_classes):
    #         for idx in range(len(self.vis_results[str(i)])):
    #             max_point = np.max(self.vis_results[str(i)][idx], axis=1) ### must be tested here
    #             thres = threshold * max_point
    #             mask = self.vis_results[str(i)][idx] > thres
    #             histogram[i][mask] += 1
    #
    #     return histogram

if __name__ == "__main__":
    pass