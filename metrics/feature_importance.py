"""
Feature Importance:
    Smart Perturbation
        1. Given only the highlighted temporal feature from vis. Methods, the other random perturbation, or
            zero pertubation, see how the Accuracy changes
        2.  Given the hightlighted temporal feature being moved in time axis, also see the Accuracy changes
"""

## -------------------
## --- Third-Party ---
## -------------------
import torch as t
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix

## -----------
## --- Own ---
## -----------
from metrics.temporal_instability import Temporal_instability
from utils import load_model

## Device Configuration
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

## Use in Datasets for UCR
class Feature_importance:
    def __init__(self, testset, histogram: np.ndarray, top_features: np.ndarray,
                 model: nn.Module):
        """
        Pertubation-based of metrics, to see the accuracy of the model in testset
        (maybe a cheating problem, when we also use pertubation-based vis. Methods)

        Parameters
        ----------
        testset: The dataset to be tested
        histogram (np.ndarray) : the histogram from temporal instability,
                            the number of times the feature be counted as importance
                            shape: [Classes, Dim, Len]
        top_features (np.ndarray) : the indexes of the top features(the most important features)
                                    , which are chosen from visualization Methods
                                    shape : [Nc, Dim, Len]
                                    Nc: number of classes
                                    Dim: Dimension of sensors(features)
                                    Len: number of featuresgroups
        model (nn.Module) : Pytorch Model to test the accuracy
        """
        self.testset = testset
        self.histogram = histogram
        self.top_features = top_features
        self.model = model

        self.num_classes = len(np.unique(self.testset.labels))
        self.criterions = {}

    # def feature_extract_per_class(self):
    #     self.threshold_class = np.zeros((self.histogram.shape[0], self.histogram.shape[1]))
    #     for i in range(self.histogram.shape[0]):
    #         for j in range(self.histogram.shape[1]):
    #             self.threshold_class[i, j] = np.max(self.histogram[i, j, :]) * self.threshold
    #
    #     mask = np.zeros(self.histogram.shape)
    #     for i in range(self.histogram.shape[0]):
    #         for j in range(self.histogram.shape[1]):
    #             mask[i, j, :] = self.histogram[i, j, :] > self.threshold_class[i, j] ## [Classes, dim, len]
    #     return mask
    def feature_masks(self, klen: float):
        """
        Parameters
        ----------
        klen (float): kernel length, the area will be perturbated
                    (Sample Length * klen)

        Returns
        -------
        mask : feature importance, the number of masks depends on Nc (Number of classes) and Ntp (Number of top features)
                Shape [Nc*Ntp, Dim, Len]
                      Len: Sample Length
        """
        pass

    def mean_perturbate(self, klen: float, importance: bool = True):
        """
        klen (float): kernel length, the area will be perturbated
                    (Sample Length * klen)

        importance: True -> perturbate the feature importance,
                    False -> perturbate the feature non-importance
        """

        self.testset_mean_perturbated = deepcopy(self.testset)
        pertubate_len = int(self.testset_mean_perturbated.data.shape[-1] * klen)

        for i in range(len(self.testset_mean_perturbated.labels)):
            for j in range(self.top_features.shape[-1]):
                mask = np.zeros(self.top_features.shape)

                cls = self.testset_mean_perturbated.labels[i]
                if importance:
                    self.testset_mean_perturbated.data[i, :, self.top_features[cls, :, (j - pertubate_len//2):(j + pertubate_len//2)]] = \
                        np.mean(self.testset.data[i, :, :], axis=-1)

                else:
                    self.testset_mean_perturbated.data[i, j, mask[self.testset_mean_perturbated.labels[i], j, :] == 0] = np.mean(
                        self.testset.data[i, j, :])
        return self.testset_mean_perturbated
    def quantile_perturbate(self, quantile = 0.8, importance: bool = True):
        """
        importance: True -> perturbate the feature importance, False -> perturbate the feature non-importance
        """
        mask = self.feature_extract_per_class()
        self.testset_quantile_perturbated = deepcopy(self.testset)
        for i in range(len(self.testset_quantile_perturbated.labels)):
            for j in range(mask.shape[1]):
                if importance:
                    self.testset_quantile_perturbated.data[i, j, mask[self.testset_quantile_perturbated.labels[i], j, :] == 1] = np.quantile(
                        self.testset.data[i, j, :], q=quantile)
                else:
                    self.testset_quantile_perturbated.data[
                        i, j, mask[self.testset_quantile_perturbated.labels[i], j, :] == 0] = np.quantile(
                        self.testset.data[i, j, :], q=quantile)
        return self.testset_quantile_perturbated
    def zero_perturbate(self, importance: bool = True):
        """
        importance: True -> perturbate the feature importance, False -> perturbate the feature non-importance
        """
        mask = self.feature_extract_per_class()
        self.testset_zero_perturbated = deepcopy(self.testset)
        for i in range(len(self.testset_zero_perturbated.labels)):
            for j in range(mask.shape[1]):
                if importance:
                    self.testset_zero_perturbated.data[i, j, mask[self.testset_zero_perturbated.labels[i], j, :] == 1] = 0
                else:
                    self.testset_zero_perturbated.data[
                        i, j, mask[self.testset_zero_perturbated.labels[i], j, :] == 0] = 0
        return self.testset_zero_perturbated
    def random_perturbate(self, importance: bool = True):
        """
        importance: True -> perturbate the feature importance, False -> perturbate the feature non-importance
        """
        mask = self.feature_extract_per_class()
        if importance:
            mask_ = (mask == 1)
        else:
            mask_ = (mask == 0)
        mask_indices = np.where(mask_)      ## mask_indices 3D = indexes of [C, D, L]
        self.testset_random_perturbated = deepcopy(self.testset)
        for i in range(len(self.testset_random_perturbated.labels)):
            for c, d, j in zip(mask_indices[0], mask_indices[1], mask_indices[2]):
                if self.testset_random_perturbated.labels[i] == c:
                    self.testset_random_perturbated.data[i, d, j] = np.random.uniform(np.min(self.testset.data[i, d, :]),
                                                                                      np.max(self.testset.data[i, d, :]))
        return self.testset_random_perturbated

    def time_shift_perturbate(self):
        mask = self.feature_extract_per_class()
        mask_indices = np.where(mask == 1)
        sequence_list = self.sequence_detect(mask_indices=mask_indices)
        self.testset_shift_perturbated = deepcopy(self.testset)
        for i in range(len(self.testset_shift_perturbated.labels)):
            for start, end in zip(sequence_list[::2], sequence_list[1::2]):
                if self.testset_shift_perturbated.labels[i] == start[0]:
                    length = end[-1] - start[-1]
                    rand_idx = np.random.randint(0, high=self.testset_shift_perturbated.data.shape[-1] - length)
                    temp = self.testset_shift_perturbated.data[i, start[1], rand_idx:rand_idx+length]
                    self.testset_shift_perturbated.data[i, start[1], rand_idx:rand_idx+length] = self.testset_shift_perturbated.data[i, start[1], start[-1]:end[-1]]
                    self.testset_shift_perturbated.data[i, start[1], start[-1]:end[-1]] = temp

        return self.testset_shift_perturbated

    def sequence_detect(self, mask_indices):
        old_idx = 0
        old_c = 0
        old_d = 0
        count = 0
        sequence_list = []
        for c, d, j in zip(mask_indices[0], mask_indices[1], mask_indices[2]):
            if old_c == c and old_d == d:
                if j - old_idx == 1:
                    if count == 0:
                        sequence_list.append([c, d, old_idx])
                    if j == mask_indices[2][-1]:
                        sequence_list.append([c, d, j])
                    count += 1
                else:
                    if count != 0:
                        sequence_list.append([c, d, old_idx])
                        count = 0
            else:
                sequence_list.append([old_c, old_d, old_idx])
                count = 0
            old_idx = j
            old_c = c
            old_d = d
        return sequence_list

    def test_accuracy(self, testset, perturbation_method: str):
        correct = 0
        predictions = None
        with t.no_grad():
            for i, (data, label) in enumerate(testset, 0): ## data size = [B, dim, len_sample]
                data = t.tensor(data).reshape((1, *data.shape))
                xt = data.float().to(device)
                label = t.tensor(label).to(device).reshape((-1, 1))  ## label should be (len, 1)
                ## Forward pass
                predicted = self.model(xt)
                predicted = t.argmax(predicted, dim=1)
                correct += (predicted == label).sum().item()
                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
        avg_acc = (correct / len(testset.labels)) * 100
        print('[Evaluation Perturbation] On testset accuracy rate: {} %'.format(avg_acc))
        predictions = np.asarray(predictions.cpu().numpy()).reshape(-1, 1)

        ## Confusion Matrix
        cm = confusion_matrix(testset.labels, predictions, labels=range(self.num_classes)) ## cm row (true labels) col (predicted labels)

        self.criterions["threshold_count_histogram"] = self.threshold
        self.criterions[f"{perturbation_method}_acc"] = avg_acc
        self.criterions[f"{perturbation_method}_confusion_matrix"] = cm
        return self.criterions




