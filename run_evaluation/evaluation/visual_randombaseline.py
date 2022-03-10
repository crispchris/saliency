## -------------------
## --- Third-Party ---
## -------------------
import sys
import os
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
from typing import List
import argparse
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from copy import deepcopy

### -----------
### --- Own ---
### -----------
from trainhelper.dataset import Dataset
from utils import read_dataset_ts, load_model, throw_out_wrong_classified
from visualize_mechanism.visual_utils import SaliencyConstructor
from models.models import FCN, TCN
from models.mlp import MLP
from models.lstm import LSTM

from visualize_mechanism.tsr import tsr_from_saliencymap

# Deep Learning Model Selection
def load_data_and_models(args):
    # Load the dataset
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    if dataset_name == "NATOPS":
        dataset = read_dataset_ts(root_dir, dataset_name, multivariate=True)
    else:
        dataset = read_dataset_ts(root_dir, dataset_name, multivariate=False)
    train_x, test_x, train_y, test_y, labels_dict = dataset[dataset_name]

    # label_summary = np.unique(list(test_y) + list(train_y))
    # num_cls = len(label_summary)

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

        testsets.append(testset)
        ## create saliency constructor
        saliency_constructor_gcs.append(SaliencyConstructor(model,
                                                            data=testset,
                                                            use_prediction=True,
                                                            device=device))
        ## add softmax to create probability
        model_softmax = nn.Sequential(model, nn.Softmax(dim=1))
        model_softmax = model_softmax.eval()
        model_softmax = model_softmax.cuda()
        model_softmaxs.append(model_softmax)

        saliency_constructors.append(SaliencyConstructor(model_softmax,
                                                         data=testset,
                                                         use_prediction=True,
                                                         device=device))

    return testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors

def get_randommaps(args,
                   testsets,
                   models,
                   model_softmaxs):
    #device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    saliencymaps = []

    for i in range(len(models)):
        ## Save the results of saliency random baseline
        random_baseline_abs = np.random.uniform(low=0.0, high=1.0, size=testsets[i].data.shape)
        random_baseline_no_abs = np.random.uniform(low=-1.0, high=1.0, size=testsets[i].data.shape)
        ## store into dictionary
        saliencymaps.append({"random_abs": random_baseline_abs,
                             "random_noabs": random_baseline_no_abs})
    return saliencymaps

def save_randommaps(args, saliencymaps):
    ### Save the Saliency Maps
    experiment_names = args.Experiments
    # experiment_names = ["experiment_11"]
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dl_selected_model = args.DLModel
    path_2_save = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"
    for i in range(len(experiment_names)):
        name = path_2_save + "randombaseline_map" + experiment_names[i] + ".npy"
        np.save(name, saliencymaps[i])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='FordB')
    parser.add_argument("--Dataset_name_save", type=str, default='FordB')
    parser.add_argument("--Experiments", nargs='+', default='experiment_7')
    parser.add_argument("--DLModel", type=str, default='TCN_withoutFC')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")
    testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors = load_data_and_models(args=args)
    
    saliencymaps = get_randommaps(args,
                                  testsets,
                                  models,
                                  model_softmaxs
                                  )
    save_randommaps(args=args,
                    saliencymaps=saliencymaps
                   )