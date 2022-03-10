"""
Explanation Intersection between the salient area and the ground truth mask over the
salient region (IoSR)
References: https://deepai.org/publication/quantitative-evaluations-on-saliency-methods-an-experimental-study
"""

## -------------------
## --- Third-Party ---
## -------------------
import os
import sys
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import argparse
import numpy as np
import torch as t

## -----------
## --- Own ---
## -----------
from utils import load_saliencies
from visual_interpretability import load_data_and_models
from visualize_mechanism.visual_utils import AggSegmentationWrapper
from metrics.pointing_game import iosaliency_regions
from pointing_game_eval import densely_labels_2_single_labels, throw_wrong_classified

def io_salient_region_scores_calcul(args,
                                    model,
                                    testset,
                                    saliency_maps: dict,
                                    targets,
                                    labels,
                                    threshold: float = 0.5,
                                    save_to_path: str = None
                                    ):
    """
    Parameters
    ----------
    args
    model: The classifier, whose saliency maps will be evaluated
    testset (Dataset): dataset with Data and Labels
    saliency_maps
    targets: the target of labels (which label in the densely label) to be evaluated
    labels: densely labels from dataset
    threshold: a threshold for saliency maps (representative if values above threshold)
    save_to_path: where the results to be saved

    Returns
    -------

    """
    root_dir = parentDir + '/../'
    # dataset_name = args.Dataset_name
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel
    experiment = args.Experiments
    # experiment = ["experiment_14"]
    experiment = experiment[0]

    ## clean out the wrong prediction labels
    clean_saliency, cleanlabels, cleantarget = throw_wrong_classified(model=model,
                                                                      dataset=testset,
                                                                      saliency_map=saliency_maps,
                                                                      target=targets,
                                                                      device=None)

    iosr_score_dict = {}
    iosr_score_dict['mean'] = {}
    iosr_score_dict['std'] = {}
    num_test_sample_dict = {}
    for key in saliency_maps.keys():
        iosr_score_dict['mean'][key], iosr_score_dict['std'][key], num_test_sample_dict[key] = iosaliency_regions(
            saliency_map=clean_saliency[key],
            targets=cleantarget,
            labels=cleanlabels,
            threhold=threshold
        )

    if save_to_path is not None:
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/"
        name_scores = path_2_save + experiment + "_" + save_to_path
        name_num_samples = path_2_save + experiment + "_number_samples_for_iosr.npy"
        np.save(name_scores, iosr_score_dict)
        np.save(name_num_samples, num_test_sample_dict)

    return iosr_score_dict

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='tool_tracking_Cluster')
    parser.add_argument("--Dataset_name_save", type=str, default='tool_tracking_Cluster')
    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='Utime')
    parser.add_argument("--Data_path", type=str, default='data/tool_tracking_data')
    parser.add_argument("--Detection", action="store_true", default=False)
    parser.add_argument("--Znorm", action="store_true", default=True)
    parser.add_argument("--One_matrix", action="store_true", default=True)
    parser.add_argument("--Sparse_labels", action="store_true", default=False)
    parser.add_argument("--Window_length", type=float, default=0.2)

    parser.add_argument("--Threshold", type=float, default=0.5)
    parser.add_argument("--Save_scores_path", type=str, default='iosr_scores.npy')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    print("Load Data and Model")
    testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors = load_data_and_models(
        args=args
    )

    ## setting
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dl_selected_model = args.DLModel
    path_2_saliency = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"
    experiments = args.Experiments
    # experiments = ["experiment_14"]
    saliency_abs_list, saliency_no_abs_list = load_saliencies(path_2_saliency, experiments)

    dl_selected_model = args.DLModel
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    save_scores_path = args.Save_scores_path

    ## for densely labels
    ## clean up targets (labels) to a single label for each sample
    segmentation_wrapper = AggSegmentationWrapper(model=models[0])

    labels_trans = densely_labels_2_single_labels(testset=testsets[0],
                                                  model=segmentation_wrapper,
                                                  device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
                                                  use_prediction=False)

    iosr_score_dict = io_salient_region_scores_calcul(args=args,
                                                      model=segmentation_wrapper,
                                                      testset=testsets[0],
                                                      saliency_maps=saliency_no_abs_list[0],
                                                      targets=labels_trans,
                                                      labels=testsets[0].labels,
                                                      threshold=args.Threshold,
                                                      save_to_path=save_scores_path)

    save_scores_path = "abs_" + save_scores_path
    iosr_score_dict_abs = io_salient_region_scores_calcul(args=args,
                                                          model=segmentation_wrapper,
                                                          testset=testsets[0],
                                                          saliency_maps=saliency_abs_list[0],
                                                          targets=labels_trans,
                                                          labels=testsets[0].labels,
                                                          threshold=args.Threshold,
                                                          save_to_path=save_scores_path)
