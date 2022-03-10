"""
Temporal Saliency Rescaling (TSR)
a Two-Step Procedure, which decouples the (time, feature) importance scores to
time and feature relevance scores

refer to: Benchmarking Deep Learning Interpretability in Time Series Predictions
https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark/tree/31ec50ec4e727dead1abad1ba8cf62eadb4de764
"""
## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import numpy as np
from typing import Dict, List
import torch as t
import torch.nn as nn
from sklearn import preprocessing

## -----------
## --- Own ---
## -----------
from visualize_mechanism.visual_utils import SaliencyConstructor
from trainhelper.dataset import Dataset

def tsr_from_saliencymap(
                         samples: np.ndarray,
                         labels: np.ndarray,
                         dl_model: nn.Module,
                         time_groups: int = 5,
                         feature_groups: int = 1,
                         threshold: float = 0.5
                         ):
    """
    TSR (Temporal Saliency Rescaling) with Time groups and Feature groups
    here only for Gradients redistribution
    modified from: https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark/tree/31ec50ec4e727dead1abad1ba8cf62eadb4de764

    Parameters
    ----------
    samples: the input data to view their saliency interpretation (shape [B, Feature, Time])
    labels: the labels of the input data (shape [B,] ... )
    dl_model: the Deep Learning Model
    time_groups: Time steps are also related to each other, how many time step in each group
    feature_groups: Feature are also related to each other, like X,Y,Z from the same sensor
    threshold: float, threshold, used in the second step, only values > threshold, which
                        will be considered in modified(improved) saliency interpretation

    Returns
    -------
    modified_saliency_rescaling
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    saliency_constructor = SaliencyConstructor(model=dl_model,
                                               data=Dataset(samples, labels),
                                               device=device)
    grad_maps = np.zeros(samples.shape)
    for idx in range(len(labels)):
        grad_maps[idx] = saliency_constructor.gradient_saliency(idx=idx,
                                                                absolute=False)

    ## two scores
    time_scores = np.zeros((samples.shape[1], samples.shape[-1]))
    feature_scores = np.zeros((samples.shape[1], samples.shape[-1]))

    ## first step: Time Axis
    time_steps_entire = False
    if samples.shape[-1] % time_groups == 0:  ## ensure time step run through every time point
        time_steps_entire = True
    num_steps = samples.shape[-1] // time_groups
    for i in range(num_steps):
        reSaliencymap = np.zeros(samples.shape)
        ## perturbate the input
        mask = samples.copy()
        means = np.expand_dims(np.mean(mask[:, :, i*time_groups:(i+1)*time_groups], axis=-1), axis=-1)
        mask[:, :, i*time_groups:(i+1)*time_groups] = means

        dataset = Dataset(mask, labels)
        mask_saliency_constructor = SaliencyConstructor(model=dl_model,
                                                        data=dataset,
                                                        device=device)
        for idx in range(len(labels)):
            reSaliencymap[idx] = mask_saliency_constructor.gradient_saliency(idx=idx,
                                                                             absolute=False)

        timeSaliency_perTimeGroup = np.absolute(grad_maps - reSaliencymap)
        score = np.sum(timeSaliency_perTimeGroup)
        time_scores[:, i * time_groups:(i+1)*time_groups] = score
    if not time_steps_entire:
        reSaliencymap = np.zeros(samples.shape)
        mask = samples.copy()
        means = np.expand_dims(
            np.mean(mask[:, :, (num_steps * time_groups):],
            axis=-1), axis=-1
        )
        mask[:, :, (num_steps * time_groups):] = means
        dataset = Dataset(mask, labels)
        mask_saliency_constructor = SaliencyConstructor(model=dl_model,
                                                        data=dataset,
                                                        device=device)
        for idx in range(len(labels)):
            reSaliencymap[idx] = mask_saliency_constructor.gradient_saliency(idx=idx,
                                                                             absolute=False)

        timeSaliency_perTimeGroup = np.absolute(grad_maps - reSaliencymap)
        score = np.sum(timeSaliency_perTimeGroup)
        time_scores[:, (num_steps * time_groups):] = score

    timeContribution = time_scores / np.max(np.abs(time_scores))
    # threshold = np.quantile(timeContribution, threshold, axis=-1)
    threshold = np.quantile(timeContribution, threshold)

    ## second step: Feature Axis
    assert samples.shape[1] % feature_groups == 0  ## ensure feature groups
    num_feature_steps = samples.shape[1] // feature_groups

    if num_feature_steps != 1:
        for i in range(num_steps):
            for j in range(num_feature_steps):
                reSaliencymap_f = np.zeros(samples.shape)
                inputmask = samples.copy()
                threshold_mask = timeContribution[0, i*time_groups:(i+1)*time_groups] > threshold
                if threshold_mask.all():
                    means = np.expand_dims(np.mean(
                        inputmask[:, j*feature_groups:(j+1)*feature_groups, i*time_groups:(i+1)*time_groups],
                        axis=-1
                    ), axis=-1
                    )
                    inputmask[:, j * feature_groups:(j + 1) * feature_groups, i * time_groups:(i + 1) * time_groups] = means
                    dataset = Dataset(inputmask, labels)
                    mask_saliency_constructor = SaliencyConstructor(model=dl_model,
                                                                    data=dataset,
                                                                    device=device)
                    for idx in range(len(labels)):
                        reSaliencymap_f[idx] = mask_saliency_constructor.gradient_saliency(idx=idx,
                                                                                         absolute=False)
                    featureSaliency_perFeatureGroup = np.absolute(grad_maps - reSaliencymap_f)
                    score = np.sum(featureSaliency_perFeatureGroup)
                    feature_scores[j*feature_groups:(j+1)*feature_groups, i*time_groups:(i+1)*time_groups] = score
                else:
                    feature_scores[j*feature_groups:(j+1)*feature_groups, i*time_groups:(i+1)*time_groups] = 0
        if not time_steps_entire:
            for j in range(num_feature_steps):
                reSaliencymap_f = np.zeros(samples.shape)
                inputmask = samples.copy()
                threshold_mask = timeContribution[0, (num_steps*time_groups):] > threshold
                if threshold_mask.all():
                    means = np.expand_dims(np.mean(
                        inputmask[:, j*feature_groups:(j+1)*feature_groups, (num_steps*time_groups):],
                        axis=-1
                    ), axis=-1
                    )
                    inputmask[:, j * feature_groups:(j + 1) * feature_groups, (num_steps * time_groups):] = means
                    dataset = Dataset(inputmask, labels)
                    mask_saliency_constructor = SaliencyConstructor(model=dl_model,
                                                                    data=dataset,
                                                                    device=device)
                    for idx in range(len(labels)):
                        reSaliencymap_f[idx] = mask_saliency_constructor.gradient_saliency(idx=idx,
                                                                                         absolute=False)
                    featureSaliency_perFeatureGroup = np.absolute(grad_maps - reSaliencymap_f)
                    score = np.sum(featureSaliency_perFeatureGroup)
                    feature_scores[j*feature_groups:(j+1)*feature_groups, (num_steps * time_groups):] = score
                else:
                    feature_scores[j*feature_groups:(j+1)*feature_groups, (num_steps * time_groups):] = 0

        featureContribution = feature_scores / np.max(np.abs(feature_scores))

        importance_score = featureContribution * timeContribution
    else:
        timeContribution[timeContribution < threshold] = 0
        importance_score = timeContribution

    return importance_score




def temporalsaliencyrescaling(saliency_method: str,
                              input: np.ndarray,
                              labels: np.ndarray,
                              dl_model: nn.Module,
                              time_steps: int = 1,
                              threshold: float = 0.6,
                              hasBaseline = None,
                              hasFeatureMask = None,
                              hasSliding_window_shapes = None):
    """
    TSR (Temporal Saliency Rescaling), improves the performance of the saliency methods

    Parameters
    ----------
    saliency_method (str) : The name of Saliency Method
                        ["grad", "ig", "sg", "gradCAM", "LRP", "GBP", "GuidedGradCAM", "LIME", "SHAP"]

    input (np.ndarray) : the input data to view their saliency interpretation (shape [B, Feature, Time])
    labels (np.ndarray) : the labels of the input data (shape [B,] ... )
    dl_model (nn.Module) : the Deep Learning Model
    time_steps (int) : steps that move in time axis
    threshold (float) : threshold, used in the second step, only values > threshold, which
                        will be considered in modified(improved) saliency interpretation

    hasBaseline : for saliency method
    hasFeatureMask : for saliency method
    hasSliding_window_shapes : for saliency method

    Returns
    -------
    modified Saliency map (np.ndarray) (shape [B, Feature, Time])
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    saliency_constructor = SaliencyConstructor(model=dl_model, data=Dataset(input, labels),
                                               device=device)
    ## two scores
    time_scores = np.zeros((input.shape[0], 1, input.shape[-1]))

    important_scores = np.zeros(input.shape)
    reSaliencymap = np.zeros(input.shape)
    normal_map = np.zeros(input.shape)
    ## normal Saliency Maps
    for idx in range(len(labels)):
        if saliency_method is "grad":
            normal_map[idx] = saliency_constructor.gradient_saliency(idx=idx)[0]
        elif saliency_method is "ig":
            normal_map[idx] = saliency_constructor.integrated_gradients(idx=idx, ig_steps=100)[0]
        elif saliency_method is "sg":
            normal_map[idx] = saliency_constructor.smooth_gradients(idx=idx, nt_samples=100, stdevs=4.0)[0]
        elif saliency_method is "gradCAM":
            normal_map[idx] = saliency_constructor.grad_cam(idx=idx,
                                          use_relu=False,
                                          attribute_to_layer_input=True)[0]
        elif saliency_method is "LRP":
            normal_map[idx] = saliency_constructor.lrp_(idx=idx)[0]
        elif saliency_method is "GBP":
            normal_map[idx] = saliency_constructor.guided_backprop(idx=idx)[0]
        elif saliency_method is "GuidedGradCAM":
            normal_map[idx] = saliency_constructor.guided_gradCAM_(idx=idx,
                                                 use_relu=False,
                                                 attribute_to_layer_input=True)[0]

    ## first step: Time Axis
    assert input.shape[-1] % time_steps == 0 ## ensure time step run through every time point
    num_steps = input.shape[-1] // time_steps
    for i in range(num_steps):
        ## perturbate the input
        mask = input.copy()

        mask[:, :, i*time_steps:(i+1)*time_steps] = 0

        dataset = Dataset(mask, labels)
        mask_saliency_constructor = SaliencyConstructor(model=dl_model,
                                                        data=dataset,
                                                        device=device)
        for idx in range(len(labels)):
            if saliency_method is "grad":
                reSaliencymap[idx] = mask_saliency_constructor.gradient_saliency(idx=idx)[0]
            elif saliency_method is "ig":
                reSaliencymap[idx] = mask_saliency_constructor.integrated_gradients(idx=idx,
                                                                                    ig_steps=50)[0]
            elif saliency_method is "sg":
                reSaliencymap[idx] = mask_saliency_constructor.smooth_gradients(idx=idx,
                                                                                nt_samples=50,
                                                                                stdevs=3.0)[0]
            elif saliency_method is "gradCAM":
                reSaliencymap[idx] = mask_saliency_constructor.grad_cam(idx=idx,
                                                                use_relu=False,
                                                                attribute_to_layer_input=True)[0]
            elif saliency_method is "LRP":
                reSaliencymap[idx] = mask_saliency_constructor.lrp_(idx=idx)[0]
            elif saliency_method is "GBP":
                reSaliencymap[idx] = mask_saliency_constructor.guided_backprop(idx=idx)[0]
            elif saliency_method is "GuidedGradCAM":
                reSaliencymap[idx] = mask_saliency_constructor.guided_gradCAM_(idx=idx,
                                                                       use_relu=False,
                                                                       attribute_to_layer_input=True)[0]
        timeSaliency_perTime = np.absolute(normal_map - reSaliencymap)
        temp = np.sum(timeSaliency_perTime, axis=-1)
        temp = np.sum(temp, axis=-1).reshape(-1, 1)
        for j in range(time_steps):
            time_scores[:, :, i*time_steps+j] = temp
    time_scores = time_scores.reshape(-1, time_scores.shape[-1])
    timeContribution = preprocessing.minmax_scale(time_scores, axis=-1)
    timeContribution = timeContribution.reshape(-1, 1, time_scores.shape[-1])
    threshold = np.quantile(timeContribution, threshold, axis=-1)

    ## Second Step: Feature Axis
    for i in range(num_steps):
        inputmask = input.copy()
        time_mask = timeContribution[:, 0, i*time_steps].reshape(-1, 1) > threshold
        mask = inputmask[time_mask.reshape(-1), :, :]
        featureContribution = np.ones((time_mask.shape[0], input.shape[1], 1)) * 0.1
        if len(mask) != 0:
            for feature in range(input.shape[1]):
                feature_saliencymap = np.zeros((mask.shape))
                #feature score
                feature_scores = np.zeros((mask.shape[0], input.shape[1], 1))
                ## perturbate
                mask[:, feature, i*time_steps:(i+1)*time_steps] = 0
                new_label = labels[time_mask.reshape(-1)]
                dataset = Dataset(mask, new_label)
                feature_saliency_constructor = SaliencyConstructor(model=dl_model,
                                                                   data=dataset,
                                                                   device=device)
                for idx in range(len(new_label)):
                    if saliency_method is "grad":
                        feature_saliencymap[idx] = feature_saliency_constructor.gradient_saliency(idx=idx)[0]
                    elif saliency_method is "ig":
                        feature_saliencymap[idx] = feature_saliency_constructor.integrated_gradients(idx=idx,
                                                                                            ig_steps=50)[0]
                    elif saliency_method is "sg":
                        feature_saliencymap[idx] = feature_saliency_constructor.smooth_gradients(idx=idx,
                                                                                        nt_samples=50,
                                                                                        stdevs=3.0)[0]
                    elif saliency_method is "gradCAM":
                        feature_saliencymap[idx] = feature_saliency_constructor.grad_cam(idx=idx,
                                                                                use_relu=False,
                                                                                attribute_to_layer_input=True)[0]
                    elif saliency_method is "LRP":
                        feature_saliencymap[idx] = feature_saliency_constructor.lrp_(idx=idx)[0]
                    elif saliency_method is "GBP":
                        feature_saliencymap[idx] = feature_saliency_constructor.guided_backprop(idx=idx)[0]
                    elif saliency_method is "GuidedGradCAM":
                        feature_saliencymap[idx] = feature_saliency_constructor.guided_gradCAM_(idx=idx,
                                                                                       use_relu=False,
                                                                                       attribute_to_layer_input=True)[0]
                featureSaliency_perFeature = np.absolute(normal_map[time_mask.reshape(-1)] - feature_saliencymap)
                temp = np.sum(featureSaliency_perFeature, axis=-1)
                temp = np.sum(temp, axis=-1).reshape(-1, 1)
                feature_scores[:, feature, :] = temp

                ## Min Max normalize
                feature_scores = feature_scores.reshape(-1, feature_scores.shape[1])
                scale_feature_scores = preprocessing.minmax_scale(feature_scores, axis=1)
                scale_feature_scores = scale_feature_scores.reshape(-1, scale_feature_scores.shape[-1], 1)

                featureContribution[time_mask.reshape(-1)] = scale_feature_scores
        for feature in range(input.shape[1]):
            score = timeContribution[:, 0, i * time_steps:(i + 1) * time_steps] * featureContribution[:, feature, 0].reshape(-1, 1)
            important_scores[:, feature, i*time_steps:(i+1)*time_steps] = score


    return important_scores



