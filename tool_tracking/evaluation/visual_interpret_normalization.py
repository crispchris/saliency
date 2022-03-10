## Mainly this file is to save the Interpretation into file (.npy)
## Used to normalize the visual interpretation
## At the end (save the interpretation as a .npy file)

## ------------------
## --- Third-Party ---
## ------------------
import os
import sys
sys.path.append('../')
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import argparse
import numpy as np

## -----------
## --- Own ---
## -----------
from visualize_mechanism.visual_utils import min_max_normalize, diverging_normalize

def load_unnorm_saliency(args):
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dl_selected_model = args.DLModel
    experiment_names = args.Experiments
    experiment_names = ["experiment_11"]

    path_2_load = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"
    saliencymaps = {}
    for i, experiment in enumerate(experiment_names):
        name = path_2_load + "saliencymaps_" + experiment + ".npy"
        # name = path_2_load + "modsaliencymaps" + experiment + ".npy"
        maps = np.load(name, allow_pickle=True)
        saliencymaps[experiment] = maps.item()
        for key in maps.item().keys():
            print(maps.item()[key].shape)
            # print(np.isnan(maps.item()[key]))
    return saliencymaps

def saliency_normalization(args, saliencymaps):
    saliencymaps_no_abs = {}
    saliencymaps_abs = {}
    experiment_names = args.Experiments
    experiment_names = ["experiment_11"]

    for i, experiment in enumerate(experiment_names):
        saliencymaps_temp_no_abs = {}
        saliencymaps_temp_abs = {}
        for key in saliencymaps[experiment].keys():
            saliencymaps_temp_no_abs[key] = np.zeros(saliencymaps[experiment][key].shape)
            saliencymaps_temp_abs[key] = np.zeros(saliencymaps[experiment][key].shape)
            for num, batch in enumerate(saliencymaps[experiment][key]):
                saliencymaps_temp_no_abs[key][num] = diverging_normalize(batch)
                saliencymaps_temp_abs[key][num] = min_max_normalize(np.absolute(batch),
                                                                    feature_range=(0, 1))
        saliencymaps_no_abs[experiment] = saliencymaps_temp_no_abs
        saliencymaps_abs[experiment] = saliencymaps_temp_abs

        for key in saliencymaps_no_abs[experiment].keys():
            for batch in saliencymaps_no_abs[experiment][key]:
                if np.isnan(batch.all()):
                    raise ValueError("There is NAN")
        for key in saliencymaps_abs[experiment].keys():
            for batch in saliencymaps_abs[experiment][key]:
                if np.isnan(batch.all()):
                    raise ValueError("There is NAN")
    return saliencymaps_abs, saliencymaps_no_abs

def save_saliency_normalization(args, saliencymaps_abs, saliencymaps_no_abs):
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dl_selected_model = args.DLModel
    experiment_names = args.Experiments
    experiment_names = ["experiment_11"]

    path_2_save = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"
    for i, experiment in enumerate(experiment_names):
        no_abs_name = path_2_save + "saliencymaps_no_abs_norm_" + experiment + ".npy"
        # no_abs_name = path_2_save + "modsaliencymaps_no_abs_norm_" + experiment + ".npy"
        np.save(no_abs_name, saliencymaps_no_abs[experiment])

        abs_name = path_2_save + "saliencymaps_abs_norm_" + experiment + ".npy"
        # abs_name = path_2_save + "modsaliencymaps_abs_norm_" + experiment + ".npy"
        np.save(abs_name, saliencymaps_abs[experiment])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset_name", type=str, default='GunPointAgeSpan_Cluster')
    parser.add_argument("--Experiments", nargs='+', default='experiment_14')
    # parser.add_argument("--Experiments", type=str, default='experiment_14')
    parser.add_argument("--DLModel", type=str, default='TCN_laststep')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")

    ## load saliency maps
    saliency_maps = load_unnorm_saliency(args=args)
    saliency_maps_abs, saliency_maps_no_abs = saliency_normalization(args=args,
                                                                     saliencymaps=saliency_maps)
    save_saliency_normalization(args=args,
                                saliencymaps_abs=saliency_maps_abs,
                                saliencymaps_no_abs=saliency_maps_no_abs)