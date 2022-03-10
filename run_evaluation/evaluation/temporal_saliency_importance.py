## This file is used to evaluate the saliency maps,
## which includes the deletion of (Un)importance/Random time series features (perturbation)
## and measure the change of the output from models

## Inspried bz Insertion/Deletion from Paper: RISE
## Reference: https://arxiv.org/abs/1806.07421v1

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
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt

### -----------
### --- Own ---
### -----------
from trainhelper.dataset import Dataset
from utils import read_dataset_ts, load_model, throw_out_wrong_classified, load_saliencies
from visualize_mechanism.visual_utils import SaliencyConstructor
from models.models import FCN, TCN
from models.lstm import LSTM
from models.mlp import MLP
from metrics.temporal_importance import TemporalImportance
from utils import clean_saliency_list, create_directory

## functions
def barplot(x, score0, score1,
            score2,
            stds: List,
            labels: List[str],
            methods: List[str],
            percent_deletion,
            replacement_method,
            model_name,
            ylabel_name,
            width=0.2,
#             typeofsaliency: str = None,
            save_to: str=None):
    """
    x: np.arange() for the length of scores
    score2: random baseline (produce a horizontal line)
    labels: for the name of scores
    methods: should have the same length as x, score0, score1
    """
    label0 = labels[0]
    label1 = labels[1]
    # label2 = labels[2]
    # plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(16, 10))
    rects1 = ax.bar(x - width, score0, width, yerr=stds[0], label=label0)
    rects2 = ax.bar(x + width, score1, width, yerr=stds[1], label=label1)
    # rects3 = ax.bar(x + 0.0, score2, width, yerr=stds[2], label=label2)

    ax.axhline(y=score2, color='r', lw=0.8, ls='--', label='Random baseline')
    ax.set_ylabel(ylabel_name, fontsize=20)
    ax.set_title(
#         f'[{model_name} with Saliency: {typeofsaliency}] {percent_deletion}% Replacement with {replacement_method} between (Un)Importance/Random Saliency',
        f'[{model_name} with Saliency: {percent_deletion}% Replacement with {replacement_method} between (Un)Importance/Random Saliency',
        fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10.0)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    # autolabel(rects3)
    if save_to is not None:
        fig.tight_layout()
#         plt.savefig(save_to + f"/temporal_importance_eval_{replacement_method}_{percent_deletion}_percent_{model_name}_with_sali_{typeofsaliency}.png")
        plt.savefig(save_to + f"/temporal_importance_eval_{replacement_method}_{percent_deletion}_percent_{model_name}.png")
        plt.close()
    else:
        fig.tight_layout()
        plt.show()

def load_data_and_models(args):
    # Load the dataset
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    if dataset_name == "NATOPS":
        dataset = read_dataset_ts(root_dir, dataset_name, multivariate=True)
    else:
        dataset = read_dataset_ts(root_dir, dataset_name, multivariate=False)
    train_x, test_x, train_y, test_y, labels_dict = dataset[dataset_name]

    ## transfer test set into Torch Dataset
    testset = Dataset(test_x, test_y)

    ## model
    models = []
    experiments = args.Experiments
    print(experiments)
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel
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
        elif report["best_epoch"][0] < 10:
            ckp_path = path_2_parameters + "checkpoints/checkpoint_00{}.ckp".format(report["best_epoch"][0])
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


def mean_phase_signal(x, data):
    ## get mean signal
    mean_signal = t.tensor(np.mean(data, axis=0).astype("float"),
                           dtype=t.float32)
    mean_signal = mean_signal.reshape(1, mean_signal.shape[0], mean_signal.shape[1])
    bmean_signal = None
    for i in range(x.shape[0]):
        if bmean_signal is None:
            bmean_signal = mean_signal
        else:
            bmean_signal = t.cat((bmean_signal, mean_signal), axis=0)
    return bmean_signal

def temporal_importance_evaluation(args,
                                   models,
                                   datasets,
                                   saliency_list,
                                   methods: List,
                                   eval_mode: str,
                                   length: float,
                                   batch_size: int,
                                   verbose: int,
                                   device,
                                   save_to: str=None,
                                   use_random_maps: bool =False):
    assert len(models) == len(datasets)
    assert len(models) == len(saliency_list)

    random_baseline = args.Random_baseline
    ## temporal importance objects
    temporal_importance_metrics = []
    for i, model in enumerate(models):
        if eval_mode in ["mean"]:
            ## a mean sample from testset
            substrate_fn = lambda x: mean_phase_signal(x, datasets[i].data)
        elif eval_mode in ["zero"]:
            substrate_fn = lambda x: t.zeros(x.shape)
        else:
            raise ValueError("evaluation mode is wrong, it should be either mean or zero")

        temporal_importance = TemporalImportance(model=model,
                                                 substrate_fn=substrate_fn,
                                                 eval_mode=eval_mode,
                                                 step=1,
                                                 device=device)
        temporal_importance_metrics.append(temporal_importance)

    auc_saliency_maps_list = []
    prediction_scores_list = []
    if random_baseline:
        auc_random_baseline_list = []
        prediction_random_list = []
    for i in range(len(models)):
        auc_saliency_maps_dict = {}
        prediction_scores_dict = {}
        if random_baseline:
            auc_random_baseline = {}
            prediction_random_baseline = {}

        batch_samples = datasets[i].data
        batch_labels = datasets[i].labels
        temporal_importance = temporal_importance_metrics[i]
        saliencymaps = saliency_list[i]
        for key in saliency_list[i].keys():
            auc_saliency_maps_dict[key] = {}
            prediction_scores_dict[key] = {}
            for key2 in ["importance", "unimportance"]:
                auc_saliency_maps_dict[key][key2] = None
                prediction_scores_dict[key][key2] = None
            print(f"[INFO] It's Method: {key}")

            if key in [list(saliency_list[i].keys())[-1]]:
                aucs, pred_scores = temporal_importance.evaluation(
                    batch_samples=batch_samples,
                    batch_labels=batch_labels,
                    batch_saliency_maps=saliencymaps[key],
                    batch_size=batch_size,
                    percent=length,
                    method=key,
                    verbose=verbose,
#                     typeofsaliency=typeofsaliency,
                    save_to=save_to,
                    random_baseline=random_baseline
                )
            else:
                aucs, pred_scores = temporal_importance.evaluation(
                    batch_samples=batch_samples,
                    batch_labels=batch_labels,
                    batch_saliency_maps=saliencymaps[key],
                    batch_size=batch_size,
                    percent=length,
                    method=key,
                    verbose=verbose,
#                     typeofsaliency=typeofsaliency,
                    save_to=save_to,
                    random_baseline=False
                )
            #### Take mean and std from model
            auc_saliency_maps_dict[key]["importance"] = aucs["importance"]
            auc_saliency_maps_dict[key]["unimportance"] = aucs["unimportance"]
            if random_baseline and key in [list(saliency_list[i].keys())[-1]]:
                auc_random_baseline["random"] = aucs["random"]
                prediction_random_baseline["random"] = pred_scores["random"]
            prediction_scores_dict[key]["importance"] = pred_scores["importance"]
            prediction_scores_dict[key]["unimportance"] = pred_scores["unimportance"]

        auc_saliency_maps_list.append(auc_saliency_maps_dict)
        prediction_scores_list.append(prediction_scores_dict)
        if random_baseline:
            auc_random_baseline_list.append(auc_random_baseline)
            prediction_random_list.append(prediction_random_baseline)
    ## mean and std
    prediction_wholetestsets_dict = {}
    auc_wholetestsets_dict = {}
    mean_auc_wholetestsets_dict = {}
    std_auc_wholetestsets_dict = {}
    if random_baseline:
        prediction_random_baseline_dict = {}
        auc_wholetestsets_random_baseline_dict = {}
        mean_auc_wholetestsets_random_baseline_dict = {}
        std_auc_wholetestsets_random_baseline_dict = {}

    for key in auc_saliency_maps_list[0].keys():
        prediction_wholetestsets_dict[key] = {}
        auc_wholetestsets_dict[key] = {}
        mean_auc_wholetestsets_dict[key] = {}
        std_auc_wholetestsets_dict[key] = {}
        import_pred = None
        unimport_pred = None
        import_auc = None
        unimport_auc = None
        if random_baseline:
            random_pred = None
            random_auc = None
        for i in range(len(auc_saliency_maps_list)): ## iter through models
            if import_auc is None:
                import_pred = prediction_scores_list[i][key]["importance"]
                unimport_pred = prediction_scores_list[i][key]["unimportance"]
                import_auc = auc_saliency_maps_list[i][key]["importance"]
                unimport_auc = auc_saliency_maps_list[i][key]["unimportance"]
                if random_baseline and key in [list(auc_saliency_maps_list[0].keys())[-1]]:
                    random_pred = prediction_random_list[i]["random"]
                    random_auc = auc_random_baseline_list[i]["random"]
            else:
                import_pred = np.concatenate((import_pred, prediction_scores_list[i][key]["importance"]),
                                             axis=1)
                unimport_pred = np.concatenate((unimport_pred, prediction_scores_list[i][key]["unimportance"]),
                                             axis=1)
                import_auc = np.concatenate((import_auc, auc_saliency_maps_list[i][key]["importance"]),
                                            axis=0)
                unimport_auc = np.concatenate((unimport_auc, auc_saliency_maps_list[i][key]["unimportance"]),
                                            axis=0)
                if random_baseline and key in [list(auc_saliency_maps_list[0].keys())[-1]]:
                    random_pred = np.concatenate((random_pred, prediction_random_list[i]["importance"]),
                                                 axis=1)
                    random_auc = np.concatenate((random_auc, auc_random_baseline_list[i]["random"]),
                                                axis=0)
        mean_auc_wholetestsets_dict[key]["importance"] = np.mean(import_auc)
        mean_auc_wholetestsets_dict[key]["unimportance"] = np.mean(unimport_auc)
        if random_baseline and key in [list(auc_saliency_maps_list[0].keys())[-1]]:
            mean_auc_wholetestsets_random_baseline_dict["random"] = np.mean(random_auc)
        std_auc_wholetestsets_dict[key]["importance"] = np.std(import_auc)
        std_auc_wholetestsets_dict[key]["unimportance"] = np.std(unimport_auc)
        if random_baseline and key in [list(auc_saliency_maps_list[0].keys())[-1]]:
            std_auc_wholetestsets_random_baseline_dict["random"] = np.std(random_auc)

        prediction_wholetestsets_dict[key]["importance"] = import_pred
        prediction_wholetestsets_dict[key]["unimportance"] = unimport_pred
        auc_wholetestsets_dict[key]["importance"] = import_auc
        auc_wholetestsets_dict[key]["unimportance"] = unimport_auc
        if random_baseline and key in [list(auc_saliency_maps_list[0].keys())[-1]]:
            auc_wholetestsets_random_baseline_dict["random"] = [random_auc]
            prediction_random_baseline_dict["random"] = [random_pred]

    if random_baseline:
        predscores_wholetestsets_dict = {'methods': prediction_wholetestsets_dict,
                                         'random': prediction_random_baseline_dict}
        auc_wholetestsets_dict = {'methods': auc_wholetestsets_dict,
                                  'random': auc_wholetestsets_random_baseline_dict}
        
        auc_wholetestsets_dict_meanstd = {'mean': mean_auc_wholetestsets_dict,
                                          'std': std_auc_wholetestsets_dict,
                                          'mean_random': mean_auc_wholetestsets_random_baseline_dict,
                                          'std_random': std_auc_wholetestsets_random_baseline_dict}
    else:
        predscores_wholetestsets_dict = {'methods': prediction_wholetestsets_dict}
        auc_wholetestsets_dict = {'methods': auc_wholetestsets_dict}
                                  
        auc_wholetestsets_dict_meanstd = {'mean': mean_auc_wholetestsets_dict,
                                          'std': std_auc_wholetestsets_dict}

    prediction_wholetestsets_df = pd.DataFrame.from_dict(predscores_wholetestsets_dict)
#     prediction_random_baseline_df = pd.DataFrame.from_dict(prediction_random_baseline_dict)
    auc_wholetestsets_df = pd.DataFrame.from_dict(auc_wholetestsets_dict)
#     auc_wholetestsets_random_baseline_df = pd.DataFrame.from_dict(auc_wholetestsets_random_baseline_dict)
    meanstd_auc_wholetestsets_df = pd.DataFrame.from_dict(auc_wholetestsets_dict_meanstd)
    
    norm_regularization = args.Regularization_norm
    norm_parameter = args.Regularization_parameter
    
    if not use_random_maps:        
        name_df_pred_nonmean = save_to + f'/{args.DLModel}_{args.Dataset_name}_{norm_regularization}_{norm_parameter}_faithfulness_ti_eval_{eval_mode}_{length}_predscores.h5'
    #     name_df_pred_nonmean_random = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_predscores_perstep_random_baseline.h5'
        name_df_nonmean = save_to + f'/{args.DLModel}_{args.Dataset_name}_{norm_regularization}_{norm_parameter}_faithfulness_ti_eval_{eval_mode}_{length}_auc.h5'
    #     name_df_nonmean_random = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_auc_random_baseline.h5'
        name_df = save_to + f'/{args.DLModel}_{args.Dataset_name}_{norm_regularization}_{norm_parameter}_faithfulness_ti_eval_{eval_mode}_{length}_auc_meanstd.h5'
    else:
        name_df_pred_nonmean = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_predscores_randombase.h5'
    #     name_df_pred_nonmean_random = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_predscores_perstep_random_baseline.h5'
        name_df_nonmean = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_auc_randombase.h5'
    #     name_df_nonmean_random = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_auc_random_baseline.h5'
        name_df = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ti_eval_{eval_mode}_{length}_auc_meanstd_randombase.h5'
        
    prediction_wholetestsets_df.to_hdf(name_df_pred_nonmean, key='df', mode='w')
#     prediction_random_baseline_df.to_hdf(name_df_pred_nonmean_random, key='df', mode='w')
    auc_wholetestsets_df.to_hdf(name_df_nonmean, key='df', mode='w')
#     auc_wholetestsets_random_baseline_df.to_hdf(name_df_nonmean_random, key='df', mode='w')
    meanstd_auc_wholetestsets_df.to_hdf(name_df, key='df', mode='w')
        
    ## plot barplot to compare between importance sequence and random sequence
    ## first comment out
    #x = np.arange(len(methods))
    #mean_scores_import = [mean_auc_wholetestsets_dict[method]["importance"] for method in methods]
    #mean_scores_unimport = [mean_auc_wholetestsets_dict[method]["unimportance"] for method in methods]
    #if random_baseline:
    #    mean_scores_random = mean_auc_wholetestsets_random_baseline_dict["random"]
    #std_scores_import = [std_auc_wholetestsets_dict[method]["importance"] for method in methods]
    #std_scores_unimport = [std_auc_wholetestsets_dict[method]["unimportance"] for method in methods]
    #if random_baseline:
    #    std_scores_random = std_auc_wholetestsets_random_baseline_dict["random"]
    #if random_baseline:
    #    barplot(x=x,
    #            score0=mean_scores_import,
    #            score1=mean_scores_unimport,
    #            score2=mean_scores_random,
    #            stds=[std_scores_import, std_scores_unimport],
    #            percent_deletion=length,
    #            model_name=args.DLModel,
    #            ylabel_name="Area under Curve of predictions",
    #            labels=["Importance", "Unimportance", 'Random'],
    #            replacement_method=args.Evaluation_mode,
    #            methods=methods,
#   #              typeofsaliency=typeofsaliency,
    #            save_to=save_to)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='GunPointAgeSpan')
    parser.add_argument("--Dataset_name_save", type=str, default='GunPointAgeSpan_Cluster')
    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='FCN_withoutFC')
    parser.add_argument("--Regularization_norm", type=str, default='dropout_regularization')
    parser.add_argument("--Regularization_parameter", type=float, default=0.2)
    
    parser.add_argument("--Evaluation_mode", type=str, default='mean')
    parser.add_argument("--Evaluation_length", type=float, default=0.15)
    parser.add_argument("--Batch_size", type=int, default=1)
    parser.add_argument("--Verbose", type=int, default=1)
    parser.add_argument("--use_randommaps", action="store_true", default=False)
#     parser.add_argument("--TypeofSaliency", type=str, default='No_abs_norm')
    parser.add_argument("--Save_to", type=str, default=None)
    parser.add_argument("--Random_baseline", action="store_true", default=False)

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
    norm_regularization = args.Regularization_norm
    regularization_parameter = args.Regularization_parameter
    experiments = args.Experiments
    
    path_2_saliency = root_dir + "results/" + dataset_name + "/" + dl_selected_model + "/" + norm_regularization + "/"
    for experiment in experiments: ## only one experiment to be passed
        if norm_regularization == "dropout_regularization":
            path_2_saliency += "dropout_{}".format(regularization_parameter) + "/" + experiment + "/"
        elif norm_regularization == "l1_regularization":
            path_2_saliency += "loss_{}".format(regularization_parameter) + "/" + experiment + "/"
        elif norm_regularization == "l2_regularization":
            path_2_saliency += "loss_{}".format(regularization_parameter) + "/" + experiment + "/"
        elif norm_regularization == "no_regularization":
            path_2_saliency += "no_regularization" + "/" + experiment + "/"
        else:
            raise ValueError("no this kind of regularization term: {}".format(norm_regularization))
    
    use_randommaps = args.use_randommaps
    filenames_appendix = [dataset_name, dl_selected_model, norm_regularization, regularization_parameter]
    saliency_lists = load_saliencies(path_2_saliency, experiments, randombaseline=use_randommaps, 
                                    filenames_appendix=filenames_appendix)
    
    ## create Directory
    temporal_importance_dir = path_2_saliency + "temporal_importance"
    directory_done = create_directory(temporal_importance_dir)
    print(f"[INFO] the Temporal Importance Directory is created at: {temporal_importance_dir}")
    
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
                       #"lrp_epsilon",
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
                       #"lrp_epsilon",
                       "lime",
                       "kernel_shap"]
            absolute_methods = ["grads", "smoothgrads", "igs"]
            # methods = ["grads",
            #            "smoothgrads",
            #            "igs",
            #            "lrp_epsilon"]
    
    saliency_dict = {}
    for method in methods:
        if use_randommaps:
            saliency_dict[method] = saliency_lists[2][0][method]
        else:
            if method in absolute_methods:
                saliency_dict[method] = saliency_lists[0][0][method]
            else:
                saliency_dict[method] = saliency_lists[1][0][method]
    saliency_list = [saliency_dict]
#     typeofsali = args.TypeofSaliency
#     if typeofsali in ["No_abs_norm"]:
#         print("No Abs is used")
#         saliency_list = saliency_no_abs_list
#     elif typeofsali in ["Abs_norm"]:
#         print("Abs norm is used")
#         saliency_list = saliency_abs_list
#     else:
#         raise ValueError("Type of saliency not found")

    saliency_list = clean_saliency_list(model_softmaxs, testset, saliency_list, cleantestsets)
    temporal_importance_evaluation(args=args,
                                   models=model_softmaxs,
                                   datasets=cleantestsets,
                                   saliency_list=saliency_list,
                                   methods=methods,
                                   eval_mode=args.Evaluation_mode,
                                   length=args.Evaluation_length,
                                   batch_size=args.Batch_size,
                                   verbose=args.Verbose,
                                   device=device,
                                   save_to=args.Save_to,
                                   use_random_maps= use_randommaps)