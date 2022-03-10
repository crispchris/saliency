"""
Explanation Pointing Game Metric
References: 
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
from metrics.pointing_game import pointing_game

def densely_labels_2_single_labels(testset,
                                   model,
                                   device,
                                   use_prediction: bool = True
                                   ):
    labels = testset.labels
    data = t.tensor(testset.data, dtype=t.float32).to(device)
    model = model.float()

    labels_trans = np.zeros((len(labels)))
    if not use_prediction:
        for i, label in enumerate(labels):
            label_summary, label_counts = np.unique(label, return_counts=True)
            ## for tool tracking dataset, we maybe don't want to see the garbage class [4]
            label_order = np.flip(np.argsort(label_counts))
            labels_trans[i] = label_summary[label_order[1]] if label_summary[label_order[0]] == 4 else label_summary[
                label_order[0]]
    else:
        predictions = model(data)
        ## for tool tracking dataset, we maybe don't want to see the garbage class [4]
        # for i in range(predictions.shape[0]):
        labels_order = np.flip(np.argsort(predictions.cpu().detach().numpy()), axis=-1)
        for i in range(len(labels)):
            labels_trans[i] = labels_order[i, 1] if labels_order[i, 0] == 4 else labels_order[i, 0]
    return t.tensor(labels_trans)

def throw_wrong_classified(model,
                           dataset,
                           saliency_map,
                           target,
                           device):
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    data = t.tensor(dataset.data, dtype=t.float32).to(device)
    labels = t.tensor(dataset.labels)
    print(f"[INFO] [before throwing wrong classified] The number of data :{len(labels)}")
    # predicted = t.zeros(labels.shape)
    mask = []
    with t.no_grad():
        i = 0
        for d, l in zip(data, labels):
            d = d.reshape((1, *d.shape))
            l = l.reshape((-1, 1))
            ## Forward pass
            prediction = model(d)

            ls, l_counts = np.unique(l.cpu().detach().numpy(), return_counts=True)
            ls_order = np.flip(np.argsort(l_counts))
            segment_order = t.flip(t.argsort(prediction, dim=1), dims=(1,)).cpu().detach().numpy()
            if ls[ls_order[0]] != 4: ## in tool tracking class 4 is garbage class (mix all other classes)
                if segment_order[0][0] == ls[ls_order[0]]:
                    mask.append(i)
                elif segment_order[0][1] == ls[ls_order[0]] and segment_order[0][0] == 4:
                    mask.append(i)
            else:
                if segment_order[0][0] == ls[ls_order[1]]:
                    mask.append(i)
                elif segment_order[0][1] == ls[ls_order[1]] and segment_order[0][0] == 4:
                    mask.append(i)
            i += 1
    cleandata = data[mask].detach().cpu().numpy()
    cleanlabels = labels[mask].detach().cpu().numpy()
    cleantarget = target[mask]
    clean_saliencymaps = {}
    for key in saliency_map.keys():
        clean_saliencymaps[key] = saliency_map[key][mask]
    print(f"[INFO] [after throwing wrong classified] The number of data :{len(cleanlabels)}")
    return clean_saliencymaps, cleanlabels, cleantarget

def pointing_game_scores_calcul(args,
                                model,
                                testset,
                                saliency_maps: dict,
                                targets,
                                labels,
                                save_to_path = None
                                ):
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

    pointing_game_scores_dict = {}
    num_test_sample_dict = {}
    for key in saliency_maps.keys():
        pointing_game_scores_dict[key], num_test_sample_dict[key] = pointing_game(saliency_map=clean_saliency[key],
                                                       targets=cleantarget,
                                                       labels=cleanlabels
                                                       )

    if save_to_path is not None:
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/"
        name_scores = path_2_save + experiment + "_" + save_to_path
        name_num_samples = path_2_save + experiment + "_number_samples_for_pointinggame.npy"
        np.save(name_scores, pointing_game_scores_dict)
        np.save(name_num_samples, num_test_sample_dict)
    return pointing_game_scores_dict




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

    parser.add_argument("--Save_scores_path", type=str, default='pointing_game_scores.npy')

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

    pointing_game_score_dict = pointing_game_scores_calcul(args=args,
                                                           model=segmentation_wrapper,
                                                           testset=testsets[0],
                                                           saliency_maps=saliency_no_abs_list[0],
                                                           targets=labels_trans,
                                                           labels=testsets[0].labels,
                                                           save_to_path=save_scores_path)

    save_scores_path = "abs_" + save_scores_path
    pointing_game_score_dict_abs = pointing_game_scores_calcul(args=args,
                                                               model=segmentation_wrapper,
                                                               testset=testsets[0],
                                                               saliency_maps=saliency_abs_list[0],
                                                               targets=labels_trans,
                                                               labels=testsets[0].labels,
                                                               save_to_path=save_scores_path)
