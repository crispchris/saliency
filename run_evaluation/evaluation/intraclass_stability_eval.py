## -------------------
## --- Third-Party ---
## -------------------
import os
import sys
sys.path.append('../')
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
from typing import List
import argparse
import torch as t
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

## -----------
## --- Own ---
## -----------
from metrics.robustness import IntermodelCheck
from utils import read_dataset_ts, load_model, throw_out_wrong_classified, load_saliencies, clean_saliency_list
from visualize_mechanism.visual_utils import SaliencyConstructor
from models.models import TCN, TCN_dense, FCN
from models.resnet import ResNet
from models.lstm import LSTM
from models.mlp import MLP
from trainhelper.dataset import Dataset, DataSplit

def load_data_and_models(args):
    # Load the dataset
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    multivariate = args.Multivariate
    dataset = read_dataset_ts(root_dir, dataset_name,
                             multivariate = multivariate)
    train_x, test_x, train_y, test_y, labels_dict = dataset[dataset_name]

    ## transfer test set into Torch Dataset
    testset = Dataset(test_x, test_y)

    ## model
    models = []
    experiments = args.Experiments
    print(experiments)
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel

    ## Load the model weights
    ## Add Softmax as the last layer to produce the probability
    testsets = []
    model_softmaxs = []
    saliency_constructors = []
    saliency_constructor_gcs = []

    for experiment in experiments:
        path_2_parameters = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/"
        report = pd.read_csv(path_2_parameters + "reports.csv")
        ## model setting and loading from checkpoint
        if int(report["best_epoch"][0]) >= 100:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_{}.ckp".format(report["best_epoch"][0])
        else:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_0{}.ckp".format(report["best_epoch"][0])

        ## Parameters
        ## Model structure
        if dl_selected_model in ["FCN_withoutFC", "FCN", "FCN_laststep"]:
            kernel_size = [int(k) for k in
                           report["kernel_size"][0][1:-1].split(',')]
            ch_out = [int(k) for k in report["Filter_numbers"][0][1:-1].split(',')]
            model = FCN(ch_in=int(testset.data.shape[1]),
                        ch_out=ch_out,
                        dropout_rate=report["dropout_rate"][0],
                        num_classes=report["num_classes"][0],
                        kernel_size=kernel_size,
                        use_fc=report["use_fc"][0],
                        use_pooling=report["use_pooling"][0],
                        input_dim=testset.data.shape)

        elif dl_selected_model in ["TCN_withoutFC", "TCN", "TCN_laststep"]:
            # dilation = [int(k) for k in report["dilation"][0][1:-1].split(',')]  ## should be always same size as ch_out
            kernel_size = [int(k) for k in
                           report["kernel_size"][0][1:-1].split(',')]  ## the size also should be the same as ch_out
            ch_out = [int(k) for k in report["Filter_numbers"][0][1:-1].split(',')]
            model = TCN(ch_in=int(testset.data.shape[1]),
                        ch_out=ch_out,
                        kernel_size=kernel_size,
                        dropout_rate=report["dropout_rate"][0],
                        use_fc=report["use_fc"][0],
                        use_pooling=report["use_pooling"][0],
                        num_classes=report["num_classes"][0],
                        input_dim=testset.data.shape)

        elif dl_selected_model in ["LSTM"]:
            hidden_size = int(report["Hidden_size"][0])
            num_layers = int(report["num_layers"][0])
            dropout = float(report["dropout_rate"][0])
            bidirectional = bool(report["bidirectional"][0])
            model = LSTM(ch_in=int(testset.data.shape[1]),
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=bidirectional,
                         num_classes=report["num_classes"][0])
            t.backends.cudnn.enabled = False

        elif dl_selected_model in ["MLP"]:
            hidden_size = [int(k) for k in report["Hidden_size"][0][1:-1].split(',')]
            dropout = [float(k) for k in report["dropout_rate"][0][1:-1].split(',')]
            model = MLP(ch_in=int(testset.data.shape[1] * testset.data.shape[-1]),
                        ch_out=hidden_size,
                        dropout_rate=dropout,
                        num_classes=report["num_classes"][0])

        model = load_model(model=model, ckp_path=ckp_path)
        models.append(model)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        ## for evaluation we need to clean datasets
        cleandata, cleanlabels = throw_out_wrong_classified(model=model, data=testset.data,
                                                            labels=testset.labels,
                                                            device=device)
        cleantestset = Dataset(cleandata, cleanlabels)

        testsets.append(cleantestset)
        ## create saliency constructor
        saliency_constructor_gcs.append(SaliencyConstructor(model,
                                                            data=cleantestset,
                                                            use_prediction=True,
                                                            device=device))
        ## add softmax to create probability
        model_softmax = nn.Sequential(model, nn.Softmax(dim=1))
        model_softmax = model_softmax.eval()
        model_softmax = model_softmax.cuda()
        model_softmaxs.append(model_softmax)

        saliency_constructors.append(SaliencyConstructor(model_softmax,
                                                         data=cleantestset,
                                                         use_prediction=True,
                                                         device=device))

    return testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors, testset

def intraclass_stability(args,
                         models,
                         datasets,
                         saliency_list,
                         methods,
                         device,
                         save_to: str = None):
    intercheck = IntermodelCheck(model=models[0],
                                 device=device)
    
    dtw_distances_dict = {}
    dtw_distances = {}

    normal_saliency = saliency_list[0]
    cleantestset = datasets[0]
    
    for key in normal_saliency.keys():
        dtw_distances_dict[key] = {}
    
    for key in normal_saliency.keys():
        dtw_distances[key] = intercheck.stability_check(saliency_maps=normal_saliency[key],
                                                       labels=cleantestset.labels,
                                                       similar_metric="dtw")
        dtw_distances_dict[key]['dtw_mean'] = dtw_distances[key][0]
        dtw_distances_dict[key]['dtw_std'] = dtw_distances[key][1]
        dtw_distances_dict[key]['dtw_classes'] = dtw_distances[key][2]
        dtw_distances_dict[key]['dtw_raw'] = dtw_distances[key][3]
        
    dtw_distances_df = pd.DataFrame.from_dict(dtw_distances_dict)
    if save_to is not None:
        dataset_name_save = args.Dataset_name_save
        dl_selected_model = args.DLModel
        experiment = args.Experiments
        root_dir = parentDir + '/../'
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment[0] + "/intraclass_stability/"
        name_scores = path_2_save + dl_selected_model + "_" + dataset_name_save + "_stability_intraclass_random_" + save_to
        dtw_distances_df.to_hdf(name_scores, key='df', mode='w')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='GunPointAgeSpan')
    parser.add_argument("--Dataset_name_save", type=str, default='GunPointAgeSpan_Cluster')
    parser.add_argument("--Multivariate", action='store_true', default=True)
    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='FCN_withoutFC')
    parser.add_argument("--Save_scores_path", type=str, default='dtw.h5')
    parser.add_argument("--use_randommaps", action="store_true", default=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")
    cleantestsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors, testset = load_data_and_models(
        args=args
    )

    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name_save
    dl_selected_model = args.DLModel
    path_2_saliency = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"
    experiments = args.Experiments
    use_randommaps = args.use_randommaps
    saliency_lists = load_saliencies(path_2_saliency, experiments, randombaseline=use_randommaps)
    # Temporal Sequence Object
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    if use_randommaps:
        methods = ["random_abs",
                  "random_noabs"]
    else:
        if dl_selected_model not in ['LSTM', 'MLP']:

            methods = ["grads",
                       "smoothgrads",
                       "igs",
                       "lrp_epsilon",
                       "gradCAM",
                       "guided_gradcam",
                       "guided_backprop",
                       "lime",
                       "kernel_shap"]
            absolute_methods = ["grads", "smoothgrads", "igs", "gradCAM"]
        else:
            methods = ["grads",
                       "smoothgrads",
                       "igs",
                       "lrp_epsilon",
                       "lime",
                       "kernel_shap"]
            absolute_methods = ["grads", "smoothgrads", "igs"]
    
    saliency_dict = {}
    for method in methods:
        if use_randommaps and method == "random_abs":
            saliency_dict["random"] = saliency_lists[2][0][method]
        elif not use_randommaps:
            if method in absolute_methods:
                saliency_dict[method] = saliency_lists[0][0][method]
            else:
                saliency_dict[method] = saliency_lists[1][0][method]
    saliency_list = [saliency_dict]
    saliency_list = clean_saliency_list(model_softmaxs, testset, saliency_list, cleantestsets)
    
    intraclass_stability(args=args,
                         models=model_softmaxs,
                         datasets=cleantestsets,
                         saliency_list=saliency_list,
                         methods=methods,
                         device=device,
                         save_to=args.Save_scores_path)