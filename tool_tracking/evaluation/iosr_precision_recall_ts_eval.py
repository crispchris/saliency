"""
Explanation Saliency maps evaluation for densely labeling
Combine Pointing Game with Precision/Recall in Time Series data
(Intersection of Salient Region) with Precision and Recall in Time series data
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
from sklearn.metrics import precision_score, recall_score, f1_score

## -----------
## --- Own ---
## -----------
from utils import load_saliencies
from visual_interpretability import load_data_and_models
from visualize_mechanism.visual_utils import AggSegmentationWrapper
from metrics.precision_recall_4_ts import TSMetric
from pointing_game_eval import densely_labels_2_single_labels, throw_wrong_classified

def saliency_maps_filter_threshold(saliency_map: np.ndarray,
                                   targets,
                                   labels,
                                   threshold: float = 0.5
                                   ):
    targets = targets.cpu().detach().numpy()
    if saliency_map.shape[0] != len(targets):
        raise ValueError("the batch size of saliency map should be the same as the length of targets")
    if saliency_map.shape[0] != len(labels):
        raise ValueError("the batch size of saliency map should be the same as the length of labels")

    labels_mask = []
    saliency_map_mask = []
    for i in range(saliency_map.shape[0]):
        mask = labels[i] == targets[i]
        mask_sum = np.sum(mask)
        if mask_sum != 0:
            max_explain = np.max(saliency_map[i, :, :])

            threshold_value = max_explain * threshold
            threshold_value_2d = [[threshold_value] * saliency_map[i, j, :].shape[-1]
                                  for j in range(saliency_map.shape[1])]
            mask_all_over_thres = saliency_map[i, :, :] > threshold_value_2d
            num_all_over_thres = np.sum(mask_all_over_thres)
            if num_all_over_thres != 0:
                mask_over_thres = np.zeros(mask_all_over_thres.shape)
                mask_over_thres[mask_all_over_thres == True] = 1
                mask_l = np.zeros(mask.shape)
                mask_l[mask == True] = 1
                labels_mask.append(mask_l)
                saliency_map_mask.append(mask_over_thres)
    print("[Number of Test samples] Number: ", len(labels_mask))
    return labels_mask, saliency_map_mask

def iosr_precision_recall_ts(args,
                             model,
                             testset,
                             saliency_maps: dict,
                             targets,
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
    #experiment = ["experiment_8"]
    experiment = experiment[0]

    ## clean out the wrong prediction labels
    clean_saliency, cleanlabels, cleantarget = throw_wrong_classified(model=model,
                                                                      dataset=testset,
                                                                      saliency_map=saliency_maps,
                                                                      target=targets,
                                                                      device=None)

    precisions_dict_classic = {}
    recalls_dict_classic = {}
    f1s_dict_classic = {}
    precisions_dict_flat = {}
    recalls_dict_flat = {}
    f1s_dict_flat_flat = {}
    # precisions_dict_front = {}
    recalls_dict_front = {}
    f1s_dict_front_flat = {}
    # precisions_dict_back = {}
    recalls_dict_back = {}
    f1s_dict_back_flat = {}
    # precisions_dict_middle = {}
    recalls_dict_middle = {}
    f1s_dict_middle_flat = {}

    for key in saliency_maps.keys():
        precisions_dict_classic[key] = {}
        recalls_dict_classic[key] = {}
        f1s_dict_classic[key] = {}
        precisions_dict_flat[key] = {}
        recalls_dict_flat[key] = {}
        f1s_dict_flat_flat[key] = {}

        recalls_dict_front[key] = {}
        f1s_dict_front_flat[key] = {}
        recalls_dict_back[key] = {}
        f1s_dict_back_flat[key] = {}
        recalls_dict_middle[key] = {}
        f1s_dict_middle_flat[key] = {}

        real_anomalies, saliency_anomalies = saliency_maps_filter_threshold(saliency_map=clean_saliency[key],
                                                                            targets=cleantarget,
                                                                            labels=cleanlabels,
                                                                            threshold=threshold)
        real_anomalies = np.asarray(real_anomalies)
        saliency_anomalies = np.asarray(saliency_anomalies)
        classic_metric_t = TSMetric(metric_option="classic",
                                  alpha_r=0.0, cardinality="one", bias_r="flat", bias_p="flat")
        tsmetric_flat = TSMetric(metric_option="time-series",
                                 alpha_r=0.0, cardinality="one", bias_r="flat", bias_p="flat")
        tsmetric_front = TSMetric(metric_option="time-series",
                                  alpha_r=0.0, cardinality="one", bias_r="front", bias_p="flat")
        tsmetric_middle = TSMetric(metric_option="time-series",
                                   alpha_r=0.0, cardinality="one", bias_r="middle", bias_p="flat")
        tsmetric_back = TSMetric(metric_option="time-series",
                                 alpha_r=0.0, cardinality="one", bias_r="back", bias_p="flat")

        ## go through every data sample
        precisions_classic_list = []
        recalls_classic_list = []
        f1s_classic_list = []
        precisions_flat_list = []
        recalls_flat_list = []
        f1s_flat_list = []
        # precisions_front_list = []
        recalls_front_list = []
        f1s_front_list = []
        # precisions_middle_list = []
        recalls_middle_list = []
        f1s_middle_list = []
        # precisions_back_list = []
        recalls_back_list = []
        f1s_back_list = []
        for i in range(real_anomalies.shape[0]):
            # for j in range(real_anomalies.shape[1]):
            saliency_anomaly = np.sum(saliency_anomalies[i], axis=0)
            saliency_anomaly[saliency_anomaly >= 1] = 1

            ## classic from sklearn
            recall_classic = recall_score(real_anomalies[i], saliency_anomaly)
            precision_classic = precision_score(real_anomalies[i], saliency_anomaly)
            f1_classic = f1_score(real_anomalies[i], saliency_anomaly)
            print("classic metric")
            print("precision: ", precision_classic, "recall: ", recall_classic, "f1: ", f1_classic)
            ## classic from time series
            precision_classic, recall_classic, f1_classic = classic_metric_t.score(
                values_real=real_anomalies[i],
                values_predicted=saliency_anomaly
            )
            print("classic metric T")
            print("precision: ", precision_classic, "recall: ", recall_classic, "f1: ", f1_classic)
            precisions_classic_list.append(precision_classic)
            recalls_classic_list.append(recall_classic)
            f1s_classic_list.append(f1_classic)
            ## flat bias position
            precision_flat, recall_flat, f1_flat = tsmetric_flat.score(
                values_real=real_anomalies[i],
                values_predicted=saliency_anomaly
            )
            print("flat metric")
            print("precision: ", precision_flat, "recall: ", recall_flat, "f1: ", f1_flat)
            precisions_flat_list.append(precision_flat)
            recalls_flat_list.append(recall_flat)
            f1s_flat_list.append(f1_flat)
            ## front bias position
            precision_front, recall_front, f1_front = tsmetric_front.score(
                values_real=real_anomalies[i],
                values_predicted=saliency_anomaly
            )
            print("front metric")
            print("precision: ", precision_front, "recall: ", recall_front, "f1: ", f1_front)
            recalls_front_list.append(recall_front)
            f1s_front_list.append(f1_front)
            ## middle bias position
            precision_middle, recall_middle, f1_middle = tsmetric_middle.score(
                values_real=real_anomalies[i],
                values_predicted=saliency_anomaly
            )
            print("middle metric")
            print("precision: ", precision_middle, "recall: ", recall_middle, "f1: ", f1_middle)
            recalls_middle_list.append(recall_middle)
            f1s_middle_list.append(f1_middle)
            ## back bias position
            precision_back, recall_back, f1_black = tsmetric_back.score(
                values_real=real_anomalies[i],
                values_predicted=saliency_anomaly
            )
            print("back metric")
            print("precision: ", precision_back, "recall: ", recall_back, "f1: ", f1_black)
            recalls_back_list.append(recall_back)
            f1s_back_list.append(f1_black)

        ## mean and std
        ## classic
        precisions_dict_classic[key]["mean"] = np.mean(precisions_classic_list)
        precisions_dict_classic[key]["std"] = np.std(precisions_classic_list)
        precisions_dict_classic[key]["raw"] = precisions_classic_list
        recalls_dict_classic[key]["mean"] = np.mean(recalls_classic_list)
        recalls_dict_classic[key]["std"] = np.std(recalls_classic_list)
        recalls_dict_classic[key]["raw"] = recalls_classic_list
        f1s_dict_classic[key]["mean"] = np.mean(f1s_classic_list)
        f1s_dict_classic[key]["std"] = np.std(f1s_classic_list)
        ## flat
        precisions_dict_flat[key]["mean"] = np.mean(precisions_flat_list)
        precisions_dict_flat[key]["std"] = np.std(precisions_flat_list)
        precisions_dict_flat[key]["raw"] = precisions_flat_list
        recalls_dict_flat[key]["mean"] = np.mean(recalls_flat_list)
        recalls_dict_flat[key]["std"] = np.std(recalls_flat_list)
        recalls_dict_flat[key]["raw"] = recalls_flat_list
        f1s_dict_flat_flat[key]["mean"] = np.mean(f1s_flat_list)
        f1s_dict_flat_flat[key]["std"] = np.std(f1s_flat_list)
        ## front
        recalls_dict_front[key]["mean"] = np.mean(recalls_front_list)
        recalls_dict_front[key]["std"] = np.std(recalls_front_list)
        recalls_dict_front[key]["raw"] = recalls_front_list
        f1s_dict_front_flat[key]["mean"] = np.mean(f1s_front_list)
        f1s_dict_front_flat[key]["std"] = np.std(f1s_front_list)
        ## middle
        recalls_dict_middle[key]["mean"] = np.mean(recalls_middle_list)
        recalls_dict_middle[key]["std"] = np.std(recalls_middle_list)
        recalls_dict_middle[key]["raw"] = recalls_middle_list
        f1s_dict_middle_flat[key]["mean"] = np.mean(f1s_middle_list)
        f1s_dict_middle_flat[key]["std"] = np.std(f1s_middle_list)
        ## back
        recalls_dict_back[key]["mean"] = np.mean(recalls_back_list)
        recalls_dict_back[key]["std"] = np.std(recalls_back_list)
        recalls_dict_back[key]["raw"] = recalls_back_list
        f1s_dict_back_flat[key]["mean"] = np.mean(f1s_back_list)
        f1s_dict_back_flat[key]["std"] = np.std(f1s_back_list)
        ## check the length
        print("the length of the samples: {}".format(len(f1s_back_list)))

    ## classic t
    classic_metric_dict = {}
    classic_metric_dict["precision"] = precisions_dict_classic
    classic_metric_dict["recall"] = recalls_dict_classic
    classic_metric_dict["f1"] = f1s_dict_classic
    ## flat
    flat_metric_dict = {}
    flat_metric_dict["precision"] = precisions_dict_flat
    flat_metric_dict["recall"] = recalls_dict_flat
    flat_metric_dict["f1"] = f1s_dict_flat_flat
    ## front
    front_metric_dict = {}
    front_metric_dict["recall"] = recalls_dict_front
    front_metric_dict["f1"] = f1s_dict_front_flat
    ## middle
    middle_metric_dict = {}
    middle_metric_dict["recall"] = recalls_dict_middle
    middle_metric_dict["f1"] = f1s_dict_middle_flat
    ## back
    back_metric_dict = {}
    back_metric_dict["recall"] = recalls_dict_back
    back_metric_dict["f1"] = f1s_dict_back_flat
    if save_to_path is not None:
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/"
        ## classic t
        name_scores_classic = path_2_save + experiment + "_classic_" + save_to_path
        np.save(name_scores_classic, classic_metric_dict)
        ## flat
        name_scores_flat = path_2_save + experiment + "_flat_" + save_to_path
        np.save(name_scores_flat, flat_metric_dict)
        ## front
        name_scores_front = path_2_save + experiment + "_front_" + save_to_path
        np.save(name_scores_front, front_metric_dict)
        ## middle
        name_scores_middle = path_2_save + experiment + "_middle_" + save_to_path
        np.save(name_scores_middle, middle_metric_dict)
        ## back
        name_scores_back = path_2_save + experiment + "_back_" + save_to_path
        np.save(name_scores_back, back_metric_dict)
    return classic_metric_dict, flat_metric_dict, front_metric_dict, middle_metric_dict, back_metric_dict


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='tool_tracking')
    parser.add_argument("--Dataset_name_save", type=str, default='tool_tracking')
    parser.add_argument("--Experiments", nargs='+', default='experiment_14')
    parser.add_argument("--DLModel", type=str, default='LSTM_dense')
    parser.add_argument("--Data_path", type=str, default='data/tool_tracking_data')
    parser.add_argument("--Detection", action="store_true", default=False)
    parser.add_argument("--Znorm", action="store_true", default=True)
    parser.add_argument("--One_matrix", action="store_true", default=True)
    parser.add_argument("--Sparse_labels", action="store_true", default=False)
    parser.add_argument("--Window_length", type=float, default=0.2)

    parser.add_argument("--Threshold", type=float, default=0.5)
    parser.add_argument("--Save_scores_path", type=str, default='abs_pr_f1_scores.npy')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    print('Load Data and Model')
    testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors = load_data_and_models(
        args=args
    )
    ## setting
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dl_selected_model = args.DLModel
    path_2_saliency = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"
    experiments = args.Experiments
    #experiments = ["experiment_8"]
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

    iosr_precision_recall_ts(args=args,
                             model=segmentation_wrapper,
                             testset=testsets[0],
                             saliency_maps=saliency_abs_list[0],
                             targets=labels_trans,
                             threshold=args.Threshold,
                             save_to_path=save_scores_path
                             )

