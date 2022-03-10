"""
Class Sensitivity:
If an explanation method is faithful, it should be giving different explanation
[For Classification] Different Classes usually have different discriminative regions, thus a good explanation method
should have clear class sensitivity

Compare the Similarity Measurement (e.g.) Pearson Correlation between the Saliency maps of the highest Class and lowest
Class
References: https://openaccess.thecvf.com/content_CVPR_2020/papers/Rebuffi_There_and_Back_Again_Revisiting_Backpropagation_Saliency_Methods_CVPR_2020_paper.pdf

"""
## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import numpy as np
import torch as t
from scipy.stats.mstats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

## -----------
## --- Own ---
## -----------
from visualize_mechanism.visual_utils import min_max_normalize, diverging_normalize

def class_sensitivity(explanation_function,
                      explanation_name: str,
                      inps,
                      max_classes,
                      min_classes,
                      **kwargs):
    """
    Compute the Pearson Correlation between the saliency maps w.r.t to output class predicted
    with highest confidence(max class) and predicted with lowest confidence(min class)

    Parameters
    ----------
    explanation_function: explanation function
    inps: input sample
        shape = [Batch, Feature, time] (Batch should be always at first)
    max_classes: the predicted class from a model (max classes) with highest confidence
    min_classes: the predicted class from a model (min classes) with lowest confidence
    kwargs: the additional features for the explanation function

    Returns
    -------
    correlation score for samples
    explanations for max class (np.ndarray)
    explanations for min class (np.ndarray)
    """
    if inps.shape[0] != len(max_classes):
        raise ValueError("Batch size between inp and max_class is wrong")
    if inps.shape[0] != len(min_classes):
        raise ValueError("Batch size between inp and max_class is wrong")

    ## Correlation scores
    correlation_scores = np.zeros((inps.shape[0]))
    ## compute the explanations
    max_explanations = np.zeros(inps.shape)
    min_explanations = np.zeros(inps.shape)
    for i, inp in enumerate(inps):
        sample = inp.reshape(1, *inp.shape)
        max_explanation = explanation_function(inp=sample,
                                               label=int(max_classes[i]),
                                               **kwargs)
        if type(max_explanation) == tuple:
            max_explanation = max_explanation[0]
    
        if explanation_name in ["grads", "smoothgrads", "igs", "gradCAM"]:
            if explanation_name in ["gradCAM"]:
                max_explanation = min_max_normalize(np.absolute(max_explanation.detach().cpu().numpy()), feature_range=(0, 1))
            else:
                max_explanation = min_max_normalize(np.absolute(max_explanation[0].detach().cpu().numpy()), feature_range=(0, 1))
        else:
            if explanation_name == "random":
                max_explanation = max_explanation.detach().cpu().numpy()
            else:
                max_explanation = diverging_normalize(max_explanation[0].detach().cpu().numpy())
        max_explanations[i] = max_explanation
        
        min_explanation = explanation_function(inp=sample,
                                               label=int(min_classes[i]),
                                               **kwargs)
        if type(min_explanation) == tuple:
            min_explanation = min_explanation[0]
        if explanation_name in ["grads", "smoothgrads", "igs", "gradCAM"]:
            if explanation_name in ["gradCAM"]:
                min_explanation = min_max_normalize(np.absolute(min_explanation.detach().cpu().numpy()), feature_range=(0, 1))
            else:
                min_explanation = min_max_normalize(np.absolute(min_explanation[0].detach().cpu().numpy()), feature_range=(0, 1))
        else:
            if explanation_name == "random":
                min_explanation = min_explanation.detach().cpu().numpy()
            else:
                min_explanation = diverging_normalize(min_explanation[0].detach().cpu().numpy())
        min_explanations[i] = min_explanation

        #spearman_corr, _ = spearmanr(min_explanations[i],
        #                             max_explanations[i],
        #                             axis=None,
        #                             nan_policy='raise'
        #                             )
        cosine_sim = cosine_similarity(min_explanations[i].reshape(1, -1),
                                       max_explanations[i].reshape(1, -1))
        correlation_scores[i] = cosine_sim
#     max_explanations = max_explanations.detach().cpu().numpy()
#     min_explanations = min_explanations.detach().cpu().numpy()
    return correlation_scores, max_explanations, min_explanations



