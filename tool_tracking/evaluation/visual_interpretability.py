## Mainly this file is to save the Interpretation into file (.npy)
## Due to the computation time from Perturbation-Based (therefore this file is created)
## Use for visual Interpretation of the Deep learning Model
## With Dataset
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
import pandas as pd
import torch as t
import torch.nn as nn

## ---------------------
## --- Tool-Tracking ---
## ---------------------
from datatools import ACC, GYR, MAG, MIC, POS, VEL

## -----------
## --- own ---
## -----------
from utils import read_dataset_ts, load_model, throw_out_wrong_classified
from visualize_mechanism.visual_utils import SaliencyConstructor, min_max_normalize
from visualize_mechanism.visual_utils import SaliencyConstructor_densely
from dataknowing.loadData import read_data_npy

from models.models import TCN, TCN_dense, FCN
from models.resnet import ResNet
from models.lstm import LSTM, LSTM_dense
from models.unet import Utime
from models.lstm_cellattention import LSTMWithInputCellAttention
from trainhelper.dataset import Dataset, DataSplit

from gun_point.evaluation.visual_interpretability import get_saliencymaps, save_saliencymaps


# Deep Learning Model Selection
def load_data_and_models(args):
    # Load the dataset
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    data_path = args.Data_path

    ## set window length and overlap
    znorm = args.Znorm  ## zero normalization
    one_matrix = args.One_matrix
    sparse_labels = args.Sparse_labels
    window_length = args.Window_length  # unit in s
    # overlap = args.Overlap  # unit in percent

    ## -----------------
    ## --- Load Data ---
    ## -----------------
    ## whole dataset from tool in data dict separately
    data_path = root_dir + '/' + data_path

    trainset, valset, testset = read_data_npy(data_path=data_path,
                                              sparse_data=sparse_labels,
                                              znorm=znorm)
    trainset = Dataset(data=trainset[0], labels=trainset[1])
    valset = Dataset(data=valset[0], labels=valset[1])
    testset = Dataset(data=testset[0], labels=testset[1])


    ## model
    models = []
    experiments = args.Experiments
    #experiments = ["experiment_8"]
    print(experiments)
    dl_selected_model = args.DLModel

    ## Load the model weights
    ## Add Softmax as the last layer to produce the probability
    testsets = []
    model_softmaxs = []
    saliency_constructors = []
    saliency_constructor_gcs = []

    for experiment in experiments:
        path_2_parameters = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/" + experiment + "/"
        report = pd.read_csv(path_2_parameters + "reports.csv")
        ## model setting and loading from checkpoint
        if int(report["best_epoch"][0]) < 100:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_0{}.ckp".format(report["best_epoch"][0])
        else:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_{}.ckp".format(report["best_epoch"][0])

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
                        input_dim=(1, *testset.data[0].shape))

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
                        input_dim=(1, *testset.data[0].shape))
        elif dl_selected_model in ["TCN_dense"]:
            kernel_size = [int(k) for k in
                           report["kernel_size"][0][1:-1].split(',')]  ## the size also should be the same as ch_out
            ch_out = [int(k) for k in report["Filter_numbers"][0][1:-1].split(',')]
            model = TCN_dense(ch_in=int(testset.data.shape[1]),
                        ch_out=ch_out,
                        kernel_size=kernel_size,
                        dropout_rate=report["dropout_rate"][0],
                        num_classes=report["num_classes"][0],
                        )

        elif dl_selected_model in ["Utime"]:
            ch_out = [int(k) for k in report["Filter_numbers"][0][1:-1].split(',')]
            kernel_size = int(report["kernel_size"][0])  ## the size also should be the same as ch_out
            maxpool_kernels = [int(k) for k in
                               report["Maxpool_kernels"][0][1:-1].split(',')]
            dilation = int(report["dilation"][0])
            model = Utime(ch_in=int(trainset.data.shape[1]),
                          ch_out=ch_out,
                          maxpool_kernels=maxpool_kernels,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          num_classes=report["num_classes"][0])

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

        elif dl_selected_model in ["LSTM_dense"]:
            hidden_size = int(report["Hidden_size"][0])
            num_layers = int(report["num_layers"][0])
            dropout = float(report["dropout_rate"][0])
            bidirectional = bool(report["bidirectional"][0])
            model = LSTM_dense(ch_in=int(testset.data.shape[1]),
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=bidirectional,
                         num_classes=report["num_classes"][0])
            t.backends.cudnn.enabled = False

        model = load_model(model=model, ckp_path=ckp_path)
        models.append(model)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        testsets.append(testset)
        ## create saliency constructor
        if sparse_labels:
            saliency_constructor_gcs.append(SaliencyConstructor(model,
                                                                data=testset,
                                                                use_prediction=True,
                                                                device=device))
        else:
            saliency_constructor_gcs.append(SaliencyConstructor_densely(model,
                                                                        data=testset,
                                                                        use_prediction=True,
                                                                        device=device))
        ## add softmax to create probability
        model_softmax = nn.Sequential(model, nn.Softmax(dim=1))
        model_softmax = model_softmax.eval()
        model_softmax = model_softmax.cuda()
        model_softmaxs.append(model_softmax)

        if sparse_labels:
            saliency_constructors.append(SaliencyConstructor(model_softmax,
                                                             data=testset,
                                                             use_prediction=True,
                                                             device=device))
        else:
            saliency_constructors.append(SaliencyConstructor_densely(model_softmax,
                                                                     data=testset,
                                                                     use_prediction=True,
                                                                     device=device))
        return testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='tool_tracking_Cluster')
    parser.add_argument("--Data_path", type=str, default='data/tool_tracking_data')
    parser.add_argument("--Detection", action="store_true", default=False)
    parser.add_argument("--Znorm", action="store_true", default=True)
    parser.add_argument("--One_matrix", action="store_true", default=True)
    parser.add_argument("--Sparse_labels", action="store_true", default=False)
    parser.add_argument("--Window_length", type=float, default=0.2)
    parser.add_argument("--Overlap", type=float, default=0.5)

    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='Utime')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")
    testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors = load_data_and_models(args=args)
    saliencymaps = get_saliencymaps(args,
                                    testsets,
                                    models,
                                    model_softmaxs,
                                    saliency_constructor_gcs,
                                    saliency_constructors)
    save_saliencymaps(args=args,
                      saliencymaps=saliencymaps)