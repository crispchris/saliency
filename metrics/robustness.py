"""
(Use No Absolute normalisation Saliency Maps)
Metric: Robustness Test
Inspired fist by location instability, by images the distances in test set
    from a point to annotated points should be similar, for filter representable. (the view of DL model)

    Extend to the Interpretation,
    (Inside the same model)
    if the interpretations for the same class are not similar -> it's not stable
    The interpretations for the same class should be similar, in order to ensure the Stability,
    which the interpretation represents the sensibility of the decision of the model (features)

    another way if the interpretations for the different class are too similar -> it's not stability

    (intra test, between Models):
     The interpretation of the same class for different models should also be similar -> Consistency

Proposed Methods: Perturbation (minimal changes) should not change the interpretation for the same class a lot
                -> Robustness
                A distance function: (similarity function) measures the change
                L1 norm, L2 norm, ... (SSIM, correlation)...

                another way if the interpretations for the different class are too similar
                -> it's not consistency (intra test)
                they should have large different..(supposely)


Inter: inside the model
With Perturbation -> Robustness check
without Perturbation -> Stability check

Intra: Between the models
without Perturbation -> Consistency check (Class-wise) -> Auf Klassen Ebene
Without Perturbation -> On the whole test set (same predictions of the samples) -> should have small distance
                        different predictions of the samples should have the large distance
                        (also kind of Consistency check)
"""

## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from pydist2.distance import pdist1
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import torch as t
import torch.nn as nn

def multi_sum(number):
    sum_til_num = 0
    for i in range(number):
        sum_til_num += i
    return sum_til_num

def minimal_perturbate(data: t.Tensor, std: float,
                       sample_shape: t.Size = t.Size([]),
                       num_perturb: int = 1):
    """
    here apply Gaussian Noise from Gaussian distribution N(0, sigma^2 * I)
    sigma is the standard deviation (we assume Mean from Gaussian Distribution is 0)
    sigma^2 is variance

    f(x) = 1/sqrt(2*pi*sigma^2) * exp(-(x)^2/2*sigma^2)

    Parameters
    ----------
    data (Tensor): the data to be perturbated (2D Tensor) [Feature, time length]
    std (float) : the standard deviation for Gaussian distribution
    sample_shape (t.Size) : from torch, the size of the sample from Gaussian distribution
                            default: t.Size([]) for only one noise to be created
                            if num_perturb is not 1, the first element should be the same as num_perturb
    num_perturb (int) : the number of new perturbated samples

    Returns
    -------
    noise (Tensor) : the gaussian noise of samples, should have shape
                    [num_perturb, feature of sample, time length of sample]
    """
    noises = np.vstack([data] * num_perturb)
    noises = t.tensor(noises)
    # noises = t.zeros((num_perturb, data.shape[0], data.shape[-1]))
    gaussian_dist = t.distributions.normal.Normal(0.0, scale=std)
    gaussian_noise = gaussian_dist.sample(sample_shape=sample_shape)
    # for i in range(num_perturb):
    #     noises[i] = data + gaussian_noise[i]
    noises += gaussian_noise
    return noises


class IntermodelCheck:
    """
    IntermodelCheck checks the interpretation inside a single DL Model
    which checks the stability and robustness

    Stability : without Perturbation (original Saliency Maps )
                Measure the distance between the interpretation inside one class
    Robustness: Add perturbation (Gaussian Noise) --> Minimal Changes
                Interpretation should not change a lot
                    --> Distance measurement defore and after
                (***) Also measure the model prediction change
                    -> if the predictions are the same
                    Interpretations should be similar
                    otherwise, the predictions aren't the same
                    Interpretations should change(not the same)
    """
    def __init__(self, model: nn.Module,
                 device=None):
        self.model = model
        self.device = device


    def stability_check(self, saliency_maps,
                        labels,
                        similar_metric: str = 'l1'):
        """
        In Stability, we want to see at the saliency inside one model
        (Recommend) Use the samples with the right classification.
            Because the wrong classification may be confused
        To see whether the distance of saliency maps inside one class stay small.

        Parameters
        ----------
        saliency_maps (np.ndarray) : Saliency Maps from a vis. Methods
                                    [#Batch, #Features, #TimeLength]
        labels (list) : the corresponding labels for samples (saliency_maps)
        similar_metric (str) : Which distance function to be used
                                ["l1","l2", "chebyshev", "spearman", "dtw"] ...
                                "spearman" : spearman correlation distance
                                "dtw": dynamic time warping

        Returns
        -------
        avg_distance (float) : average distance of the saliency maps
        std_distance (float) : standard deviation of the saliency maps
        """
        ## check out the number of classes
        classes = np.unique(labels)
        num_cls = len(classes)
        if similar_metric is "dtw":
            class_distances = np.zeros((num_cls, 1))
            class_distances_raw = {}
        else:
            class_distances = np.zeros((num_cls, saliency_maps.shape[1]))
        for cls in classes:
            idx = [labels == cls]
            ## shape [#B, #F, #Len]
            saliency_map = saliency_maps[idx]
            distances = np.zeros((saliency_map.shape[1], multi_sum(saliency_map.shape[0])))
            saliency_maps_1 = np.swapaxes(saliency_map, axis1=1, axis2=2) ## [#batch, #time, #features]
            distance = np.zeros((saliency_map.shape[0], saliency_map.shape[0]))
            distance_dtw = np.zeros((saliency_map.shape[0], saliency_map.shape[0]))
            
            for i in range(saliency_maps_1.shape[-1]):
                if similar_metric is "l1":
                    distance = pdist1(saliency_maps_1[:, :, i], "cityblock")
                elif similar_metric is "l2":
                    distance = pdist1(saliency_maps_1[:, :, i], "euclidean")
                elif similar_metric is "chebyshev":
                    distance = pdist1(saliency_maps_1[:, :, i], "chebyshev")
                elif similar_metric is "spearman":
                    distance = pdist1(saliency_maps_1[:, :, i], "spearman")
                elif similar_metric is "dtw":
                    distance_dtw += dtw.distance_matrix_fast(saliency_maps_1[:, :, i].astype(np.double),
                                                             window=20
                                                            )

                if similar_metric not in ["dtw"]:
                    distances[i] = distance
            if similar_metric is "dtw":
                class_distances[cls] = np.mean(distance_dtw)  ## per class
                class_distances_raw[str(cls)] = distance_dtw  ## through all features
            else:
                class_distances[cls] = np.mean(distances, axis=-1) ## mean over the samples

        ## compute mean, std of distances (from features)
        if similar_metric is "dtw":
            intra_class_mean_distance = np.mean(class_distances) ## per visual method
            intra_class_std_distance = np.std(class_distances)
        else:
            inter_feature_mean = np.mean(class_distances, axis=-1)  ## mean over feature axis, get the distance per class
            intra_class_mean_distance = np.mean(inter_feature_mean)
            intra_class_std_distance = np.std(inter_feature_mean)

        return intra_class_mean_distance, intra_class_std_distance, class_distances, class_distances_raw

    def robustness_check(self, samples,
                         labels,
                         saliency_maps,
                         saliency_method,
                         similar_metric: str = 'l1',
                         num_perturb: int = 20,
                         noise_std: float = 0.01,
                         absolute: bool = True
                         ):
        """
        In Robustness, we want to see at the saliency inside one model
        (Recommend) Use the samples with the right classification.
            Because the wrong classification may be confused
        To see whether the distance of saliency maps inside one class stay small,
        if we pertubate them (minimal change)

        Warnings: May take a lot of compute times

        Parameters
        ----------
        samples (np.ndarray): the sample on the testset (used the right classified samples)
        labels (list) : the corresponding labels for samples (saliency_maps)
        saliency_maps (np.ndarray) : Saliency Maps from a vis. Methods
                                    [#Batch, #Features, #TimeLength]
        saliency_method (object) : a Saliency method to be used for evalution
                                    reproduce the saliency maps for perturbated data

        similar_metric (str) : Which distance function to be used
                                ["l1","l2", "chebyshev"] ...
        num_perturb (int) : the number of times that samples should be perturbated
        noise_std (float) : the standard deviation for Gaussian distribution
        absolute (bool) : Whether using absolute value for saliency map
        Returns
        -------
        avg_distance (float) : average distance of the saliency maps
        std_distance (float) : standard deviation of the saliency maps
        """
        ## perturbation
        for i in range(samples.shape[0]):
            noises = minimal_perturbate(data=samples[i],
                                        std=noise_std,
                                        sample_shape=t.Size([num_perturb, 1]),
                                        num_perturb=num_perturb)
            ### model pass
            probs = self.model(noises.to(self.device))
            predicts = t.argmax(probs, dim=1)
            right_idxs = [predicts == labels[i]]
            wrong_idxs = [predicts != labels[i]]
            right_noises = noises[right_idxs]
            wrong_noises = noises[wrong_idxs]
            ### saliency
            ## right saliency maps
            r_maps = t.zeros(right_noises.shape)
            for j in range(right_noises.shape[0]):
               r_maps[j] = saliency_method.attribute(input=right_noises[j],
                                                     target=labels[i],
                                                     abs=absolute)
            w_maps = t.zeros(wrong_noises.shape)
            for j in range(wrong_noises.shape[0]):
                w_maps[j] = saliency_method.attribute(input=wrong_noises[j],
                                                      target=labels[i],
                                                      abs=absolute)
            ### distance
            ### in right maps, calculate the distance between
            ### saliency_maps[i] and r_maps
            ### otherwise,
            ### saliency_maps[i] and w_maps






