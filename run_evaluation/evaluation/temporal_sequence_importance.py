## This file is used to evaluate the saliency maps, which includes the gap scores in deletion(swap or mean
## or zero) of importance continuous sequences
## Reference: https://arxiv.org/abs/1909.07082

## import libraries
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
import matplotlib.pyplot as plt

### -----------
### --- Own ---
### -----------
from utils import load_saliencies, clean_saliency_list, create_directory
from temporal_saliency_importance import load_data_and_models
from metrics.temporal_sequence_eval import TemporalSequenceEval

def barplot(x, score0, score1,
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
    labels: for the name of scores
    methods: should have the same length as x, score0, score1
    """
    label0 = labels[0]
    label1 = labels[1]
    # plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(16, 10))
    rects1 = ax.bar(x - width / 2, score0, width, yerr=stds[0], label=label0)
    rects2 = ax.bar(x + width / 2, score1, width, yerr=stds[1], label=label1)

    ax.set_ylabel(ylabel_name, fontsize=20)
    ax.set_title(
#         f'[{model_name} with Saliency: {typeofsaliency}] {percent_deletion}% Replacement with {replacement_method} between Importance/Random Saliency',
        f'[{model_name}] {percent_deletion}% Replacement with {replacement_method} between Importance/Random Saliency',
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
    if save_to is not None:
        fig.tight_layout()
#         plt.savefig(save_to + f"/time_sequence_eval_{replacement_method}_{percent_deletion}_percent_{model_name}_with_sali_{typeofsaliency}.png")
        plt.savefig(save_to + f"/time_sequence_eval_{replacement_method}_{percent_deletion}_percent_{model_name}.png")
        plt.close()
    else:
        fig.tight_layout()
        plt.show()

def temporal_sequence_evaluate(args,
                               models,
                               datasets,
                               saliency_list,
#                                typeofsaliency,
                               methods: List,
                               eval_mode: str,
                               length: float,
                               batch_size: int,
                               verbose: int,
                               device,
                               save_to: str=None,
                               use_random_maps: bool= True):
    assert len(models) == len(datasets)
    assert len(models) == len(saliency_list)
    temporal_sequence_metrics = []

    for model in models:
        temporal_sequence = TemporalSequenceEval(model=model,
                                                 eval_mode=eval_mode,
                                                 length=length,
                                                 device=device)
        temporal_sequence_metrics.append(temporal_sequence)
    ## Evaluation
    ## whole testset
    gap_scores_wholetestsets_list = []
    ran_gap_scores_wholetestsets_list = []

    for i in range(len(models)):
        gap_scores_methods_dict = {}
        ran_gap_scores_methods_dict = {}

        batch_samples = datasets[i].data
        batch_labels = datasets[i].labels
        temporal_sequence = temporal_sequence_metrics[i]
        normal_saliency = saliency_list[i]
        # for method in methods:
        for key in normal_saliency.keys():
            print(f"[INFO] It's method: {key}")
            gap_scores_methods_dict[key], ran_gap_scores_methods_dict[key] = temporal_sequence.evaluation(
                batch_samples=batch_samples,
                batch_labels=batch_labels,
                batch_saliency_maps=normal_saliency[key],
                batch_size=batch_size,
                verbose=verbose,
                method=key,
#                 typeofSaliency=typeofsaliency,
                save_to=save_to
            )
        gap_scores_wholetestsets_list.append(gap_scores_methods_dict)
        ran_gap_scores_wholetestsets_list.append(ran_gap_scores_methods_dict)

    ## mean and std for plot
    mean_gap_scores_wholetestsets_dict = {}
    ran_mean_gap_scores_wholetestsets_dict = {}
    std_gap_scores_wholetestsets_dict = {}
    ran_std_gap_scores_wholetestsets_dict = {}
    gap_scores_wholetestsets_dict = {}
    ran_gap_scores_wholetestsets_dict = {}

    for key in gap_scores_wholetestsets_list[0].keys():
        scores = None
        ran_scores = None
        for i in range(len(models)):
            if scores is None:
                scores = gap_scores_wholetestsets_list[i][key]
                ran_scores = ran_gap_scores_wholetestsets_list[i][key]
            else:
                scores = np.concatenate((scores, gap_scores_wholetestsets_list[i][key]), axis=0)
                ran_scores = np.concatenate((ran_scores, ran_gap_scores_wholetestsets_list[i][key]), axis=0)

        mean_gap_scores_wholetestsets_dict[key] = np.mean(scores)
        ran_mean_gap_scores_wholetestsets_dict[key] = np.mean(ran_scores)
        std_gap_scores_wholetestsets_dict[key] = np.std(scores)
        ran_std_gap_scores_wholetestsets_dict[key] = np.std(ran_scores)

        gap_scores_wholetestsets_dict[key] = scores
        ran_gap_scores_wholetestsets_dict[key] = ran_scores
    scores_wholetestsets_dict_nonmean = {'importance': gap_scores_wholetestsets_dict,
                                         'random': ran_gap_scores_wholetestsets_dict}
    scores_wholetestsets_dict = {'importance_mean': mean_gap_scores_wholetestsets_dict,
                                 'importance_std': std_gap_scores_wholetestsets_dict,
                                 'random_mean': ran_mean_gap_scores_wholetestsets_dict,
                                 'random_std': ran_std_gap_scores_wholetestsets_dict}
    scores_wholetestsets_df_nonmean = pd.DataFrame.from_dict(scores_wholetestsets_dict_nonmean)
    scores_wholetestsets_df = pd.DataFrame.from_dict(scores_wholetestsets_dict)
    
    norm_regularization = args.Regularization_norm
    norm_parameter = args.Regularization_parameter
    
    if not use_random_maps:
        name_df_nonmean = save_to + f'/{args.DLModel}_{args.Dataset_name}_{norm_regularization}_{norm_parameter}_faithfulness_ts_eval_{eval_mode}_{length}_gap.h5'
        name_df = save_to + f'/{args.DLModel}_{args.Dataset_name}_{norm_regularization}_{norm_parameter}_faithfulness_ts_eval_{eval_mode}_{length}_gap_meanstd.h5'
    else:
        name_df_nonmean = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ts_eval_{eval_mode}_{length}_gap_randombase.h5'
        name_df = save_to + f'/{args.DLModel}_{args.Dataset_name}_faithfulness_ts_eval_{eval_mode}_{length}_gap_meanstd_randombase.h5'
    scores_wholetestsets_df_nonmean.to_hdf(name_df_nonmean, key='df', mode='w')
    scores_wholetestsets_df.to_hdf(name_df, key='df', mode='w')

    ## plot barplot to compare between importance sequence and random sequence
#    x = np.arange(len(methods))
#    mean_gap_scores = [mean_gap_scores_wholetestsets_dict[key] for key in gap_scores_wholetestsets_list[0].keys()]
#    ran_mean_gap_scores = [ran_mean_gap_scores_wholetestsets_dict[key] for key in gap_scores_wholetestsets_list[0].keys()]
#    std_gap_scores = [std_gap_scores_wholetestsets_dict[key] for key in gap_scores_wholetestsets_list[0].keys()]
#    ran_std_gap_scores = [ran_std_gap_scores_wholetestsets_dict[key] for key in gap_scores_wholetestsets_list[0].keys()]

#    barplot(x=x,
#            score0=mean_gap_scores,
#            score1=ran_mean_gap_scores,
#            stds=[std_gap_scores, ran_std_gap_scores],
#            percent_deletion=length,
#            model_name=args.DLModel,
#            ylabel_name="Gap Scores between origin and modified",
#            labels=["Importance", "Random"],
#            replacement_method=args.Evaluation_mode,
#            methods=list(gap_scores_wholetestsets_list[0].keys()),
##             typeofsaliency=typeofsaliency,
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
    
    parser.add_argument("--Evaluation_mode", type=str, default='swap')
    parser.add_argument("--Evaluation_length", type=float, default=0.15)
    parser.add_argument("--Batch_size", type=int, default=1)
    parser.add_argument("--Verbose", type=int, default=1)
    parser.add_argument("--use_randommaps", action="store_true", default=False)
#     parser.add_argument("--TypeofSaliency", type=str, default='No_abs_norm')
    parser.add_argument("--Save_to", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")
    cleantestsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors, testset = load_data_and_models(
        args=args
    )

    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dataset_name_save = args.Dataset_name_save
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
    
    ## create Directory (For Saving files)
    temporal_sequence_dir = path_2_saliency + "temporal_sequence"
    directory_done = create_directory(temporal_sequence_dir)
    print(f"[INFO] the Temporal Importance Directory is created at: {temporal_sequence_dir}")
    
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
#         print("No Abs norm is used")
#         saliency_list = saliency_no_abs_list
#     elif typeofsali in ["Abs_norm"]:
#         print("Abs norm is used")
#         saliency_list = saliency_abs_list
#     else:
#         raise ValueError("Type of saliency not found")

    saliency_list = clean_saliency_list(model_softmaxs, testset, saliency_list, cleantestsets)
    temporal_sequence_evaluate(args=args,
                               models=model_softmaxs,
                               datasets=cleantestsets,
                               saliency_list=saliency_list,
#                                typeofsaliency=typeofsali,
                               methods=methods,
                               eval_mode=args.Evaluation_mode,
                               length=args.Evaluation_length,
                               batch_size=args.Batch_size,
                               verbose=args.Verbose,
                               device=device,
                               save_to=args.Save_to,
                               use_random_maps= use_randommaps)