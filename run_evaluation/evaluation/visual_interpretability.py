## Mainly this file is to save the Interpretation into file (.npy)
## Due to the computation time from Perturbation-Based (therefore this file is created)
## Use for visual Interpretation of the Deep learning Model
## With Dataset
## At the end (save the interpretation as a .npy file)

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
#from visualize_mechanism.visual_func import SaliencyFunctions
#from visualize_mechanism.lrp import LRP_individual
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
    
    ## norm
    norm_regularization = args.Regularization_norm
    regularization_parameter = args.Regularization_parameter
    
    ## Load the model weights
    ## Add Softmax as the last layer to produce the probability
    testsets = []
    model_softmaxs = []
    saliency_constructors = []
    saliency_constructor_gcs = []

    for experiment in experiments:
        path_2_parameters = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + norm_regularization + "/"
        if norm_regularization == "dropout_regularization":
            path_2_parameters += "dropout_{}".format(regularization_parameter) + "/" + experiment + "/"
        elif norm_regularization == "l1_regularization":
            path_2_parameters += "loss_{}".format(regularization_parameter) + "/" + experiment + "/"
        elif norm_regularization == "l2_regularization":
            path_2_parameters += "loss_{}".format(regularization_parameter) + "/" + experiment + "/"
        elif norm_regularization == "no_regularization":
            path_2_parameters += "no_regularization" + "/" + experiment + "/"
        else:
            raise ValueError("no this kind of regularization term: {}".format(norm_regularization))

        report = pd.read_csv(path_2_parameters + "reports.csv")
        ## model setting and loading from checkpoint
        if int(report["best_epoch"][0]) >= 100:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_{}.ckp".format(report["best_epoch"][0])
        else:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_0{}.ckp".format(report["best_epoch"][0])
        print(f"load from {ckp_path}")
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

def get_saliencyrescaler(args,
                         testset,
                         model,
                         time_groups,
                         feature_groups=1,
                         quantile_threshold=0.5,
                         save: bool = False
                         ):
    rescaledSaliency = tsr_from_saliencymap(
        samples=testset.data,
        labels=testset.labels,
        dl_model=model,
        time_groups=time_groups,
        feature_groups=feature_groups,
        threshold=quantile_threshold
    )
    if save:
        experiment_names = args.Experiments
        experiment_names = experiment_names[0]
        
        root_dir = parentDir + '/../'
        dataset_name = args.Dataset_name_save
        dl_selected_model = args.DLModel
        path_2_save = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/"

        name = path_2_save + "rescaled_grads_" + experiment_names + ".npy"
        np.save(name, rescaledSaliency)
    return rescaledSaliency

def get_saliencymaps(args,
                     testsets,
                     models,
                     model_softmaxs,
                     saliency_constructor_gcs,
                     saliency_constructors):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    saliencymaps = []

    for i in range(len(models)):
        saliency_constructor = saliency_constructors[i]
        saliency_constructor_gc = saliency_constructor_gcs[i]

        ## Save the results of saliency
        grads = np.zeros(testsets[i].data.shape)
        igs = deepcopy(grads)
        smoothgrads = deepcopy(grads)
        lime_maps = deepcopy(grads)
        shap_maps = deepcopy(grads)
        ENABLE_LRP_EPSION = False
        if ENABLE_LRP_EPSION:
            lrp_epsilon_maps = deepcopy(grads)
        if args.DLModel not in ["LSTM", "MLP", "LSTM_dense"]:
            gradCam_maps = deepcopy(grads)
            g_gradcam_maps = deepcopy(grads)
            gbp_maps = deepcopy(grads)
        # if args.DLModel not in ["LSTM", "LSTM_dense"]:
        #     lrp_gamma_maps = deepcopy(grads)

        for idx in range(len(testsets[i].labels)):
        #     print("Label of sample: {}".format(cleantestset.labels[idx]))
            ## Gradient Based
            grads[idx] = saliency_constructor_gc.gradient_saliency(idx=idx, absolute=False)[0]
            igs[idx] = saliency_constructor_gc.integrated_gradients(idx=idx, ig_steps=60, absolute=False)[0]
            smoothgrads[idx] = saliency_constructor_gc.smooth_gradients(idx=idx,
                                                                     nt_samples=60,
                                                                     stdevs=0.1,
                                                                     absolute=False)[0]

            if ENABLE_LRP_EPSION:
                if args.DLModel in ["LSTM", "LSTM_dense"]:
                    lrp_epsilon_maps[idx] = saliency_constructor_gc.lrp4lstm_(idx=idx,
                                                                              absolute=False)
                else:
                    lrp_epsilon_maps[idx] = saliency_constructor_gc.lrp_(idx=idx,
                                                                         rule="epsilon",
                                                                         absolute=False)[0]
            if args.DLModel not in ["LSTM", "MLP", "LSTM_dense"]:
                gradCam_maps[idx] = saliency_constructor_gc.grad_cam(idx=idx,
                                                                     use_relu=True,
                                                                     layer_to_grad="gap_softmax.conv1",
                                                                     attribute_to_layer_input=True,
                                                                     absolute=False)[0]
                g_gradcam_maps[idx] = saliency_constructor_gc.guided_gradCAM_(idx=idx,
                                                                              use_relu=True,
                                                                              layer_to_grad="gap_softmax.conv1",
                                                                              attribute_to_layer_input=True,
                                                                              absolute=False)[0]
                gbp_maps[idx] = saliency_constructor_gc.guided_backprop(idx=idx,
                                                                        absolute=False)[0]

            ### Perturbation Based
            lime_maps[idx] = saliency_constructor_gc.lime_(idx=idx,
                                                           num_features=125,
                                                           n_sample=800,
                                                           baseline="total_mean",
                                                           kernel_width=5.0,
                                                           absolute=False)[0]

            shap_maps[idx] = saliency_constructor_gc.kernelshap_(idx=idx,
                                                                  n_sample=800,
                                                                  baseline="total_mean",
                                                                  num_features=125,
                                                                  absolute=False)[0]
        ## check the model accuracy
        normal_model_acc = saliency_constructor.get_model_accuracy()
        normal_model_acc_gc = saliency_constructor_gc.get_model_accuracy()

        ## store into dictionary
        if ENABLE_LRP_EPSION:
            if args.DLModel not in ["LSTM", "MLP"]:
                saliencymaps.append({"grads": grads,
                                     "smoothgrads": smoothgrads,
                                     "igs": igs,
                                     "lrp_epsilon": lrp_epsilon_maps,
                                     # "lrp_gamma": lrp_gamma_maps,
                                     "gradCAM": gradCam_maps,
                                     "guided_gradcam": g_gradcam_maps,
                                     "guided_backprop": gbp_maps,
                                     "lime": lime_maps,
                                     "kernel_shap": shap_maps})
            #     saliencymaps.append({'lrp_epsilon': lrp_epsilon_maps})
            else:
                saliencymaps.append({"grads": grads,
                                     "smoothgrads": smoothgrads,
                                     "igs": igs,
                                     "lrp_epsilon": lrp_epsilon_maps,
                                     "lime": lime_maps,
                                     "kernel_shap": shap_maps})
        else:
            if args.DLModel not in ["LSTM", "MLP"]:
                saliencymaps.append({"grads": grads,
                                     "smoothgrads": smoothgrads,
                                     "igs": igs,
                                     # "lrp_gamma": lrp_gamma_maps,
                                     "gradCAM": gradCam_maps,
                                     "guided_gradcam": g_gradcam_maps,
                                     "guided_backprop": gbp_maps,
                                     "lime": lime_maps,
                                     "kernel_shap": shap_maps})
            #     saliencymaps.append({'lrp_epsilon': lrp_epsilon_maps})
            else:
                saliencymaps.append({"grads": grads,
                                     "smoothgrads": smoothgrads,
                                     "igs": igs,
                                     "lime": lime_maps,
                                     "kernel_shap": shap_maps})

    return saliencymaps

def save_saliencymaps(args, saliencymaps):
    ### Save the Saliency Maps
    experiment_names = args.Experiments
    # experiment_names = ["experiment_11"]
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dl_selected_model = args.DLModel
    norm_regularization = args.Regularization_norm
    regularization_parameter = args.Regularization_parameter
    
    path_2_save = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/" + norm_regularization + "/"
    
    for i in range(len(experiment_names)):
        if norm_regularization == "dropout_regularization":
            path_2_save += "dropout_{}".format(regularization_parameter) + "/" + experiment_names[i] + "/"
        elif norm_regularization == "l1_regularization":
            path_2_save += "loss_{}".format(regularization_parameter) + "/" + experiment_names[i] + "/"
        elif norm_regularization == "l2_regularization":
            path_2_save += "loss_{}".format(regularization_parameter) + "/" + experiment_names[i] + "/"
        elif norm_regularization == "no_regularization":
            path_2_save += "no_regularization" + "/" + experiment_names[i] + "/"
        else:
            raise ValueError("no this kind of regularization term: {}".format(norm_regularization))
            
        name = path_2_save + "saliencymaps_{}_{}_{}_{}".format(dataset_name, dl_selected_model, norm_regularization, regularization_parameter) + experiment_names[i] + ".npy"
        print(f"save to {name}")
        np.save(name, saliencymaps[i])

def load_saliencymaps(path2folder, experiments: list):
    saliencymaps = []
    for i in range(len(experiments)):
        # name = path2folder + "lrpmaps_" + experiments[i] + ".npy"
        name = path2folder + "saliencymaps_" + experiments[i] + ".npy"
        maps = np.load(name, allow_pickle=True)
        saliencymaps.append(maps.item())
    return saliencymaps

def test_args(*args):
    list = [item for item in args]
    return list

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='Ford')
    parser.add_argument("--Dataset_name_save", type=str, default='Ford')
    parser.add_argument("--Experiments", nargs='+', default='experiment_7')
    parser.add_argument("--DLModel", type=str, default='TCN_laststep')
    parser.add_argument("--Regularization_norm", type=str, default='dropout_regularization')
    parser.add_argument("--Regularization_parameter", type=float, default=0.2)
    

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

    # rescaled_grads = get_saliencyrescaler(args=args,
    #                                       testset=testsets[0],
    #                                       model=models[0],
    #                                       time_groups=5,
    #                                       feature_groups=1,
    #                                       quantile_threshold=0.5,
    #                                       save=True)


