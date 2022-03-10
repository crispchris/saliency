"""
Result reader
"""

import numpy as np
import itertools

from pandas.core.algorithms import isin, mode

# import h5py
import pandas as pd
import torch as t
import torch.nn as nn

import importlib
import pickle


import os


def read_results(filename):
    if not os.path.exists(filename):
        # print(f"Did not find {filename}.")
        return None

    # Results are saved as an array that contains one dictionary
    return np.load(filename, allow_pickle=True).item()


def read_h5(filename):
    if not os.path.exists(filename):
        # print(f"Did not find {filename}.")
        return None
    dataframe = pd.read_hdf(filename)
    return dataframe


BASEDIR = "../results/"
DATASETS = ["FordA", "FordB", "NATOPS"]  # "tooltracking", "gunpointagespan"
MODELS = [
    "FCN_withoutFC"
]  # "TCN_laststep"# , "FCN", "TCN", "Utime", "LSTM_dense", # "MLP", "FCN", "TCN"
PERSPECTIVES = ["faithfulness_ti", "faithfulness_ts", "accuracy"]
# PERSPECTIVES = ["sanity", "faithfulness_ti", "faithfulness_ts", "sensitivity", "robustness", "intraclassstability"] #  "localization", "faithfulness_ti"
VISUALIZATIONS = [
    "grads",
    "smoothgrads",
    "igs",
    "lrp_epsilon",
    "gradCAM",
    "guided_gradcam",
    "guided_backprop",
    "lime",
    "kernel_shap",
]

REGULARIZATIONS = ["dropout", "l1", "l2", "no"]
REGUL_PARAS = [0.2, 0.5, 0.001, 0.01, 0.1, 0.375, 0.03, 0.05, 0.005]

# VISUALIZATIONSRANDOM = ["random"]
# MODELSRANDOM = ["TCN_withoutFC"]
# DATASETSRANDOM = ["FordB"]

# Prev: "gradcam", "lrp", "integratedgradients", "gradient", "lime"
# METRICS = ["class_sensitivity", "class_stability", "sanity_check", "sensitivity_check", "temporal_importance", "temporal_sequence"]


# We need to gather the following results:
#
# Faithfulness:
#   temporal_importance/FCN_withoutFC_FordA_faithfulness_ti_eval_mean_0.2_auc.h5
#   temporal_sequence/FCN_withoutFC_FordA_faithfulness_ts_eval_{mean/swap/zero}_0.2_gap.h5
# Sensitivity:
#   class_sensitivity/FCN_withoutFC_FordA_sensitivity_cs_correlation.npy
# Stability:
#   sensitivity_max/FCN_withoutFC_FordA_stability_sm_scores.npy?
# Sanity:
#   sanity_checkFCN_withoutFC_FordA_sanity_Cascade_random_accuracy_layers.npy
# or sanity_checkFCN_withoutFC_FordA_sanity_Cascade_random_saliencymaps_{fcn.0.conv1_abs}.npy
# Localization:
#   (not yet there).


def negate_all(dict_with_values):
    # print(dict_with_values)
    return {k: -v for k, v in dict_with_values.items()}


def normalize_robustness(complete_dataframe):
    # We remove the robustness data, process it, and concatenate afterwards
    robustness_data = complete_dataframe.loc[
        complete_dataframe.perspective == "robustness"
    ]
    # Normalize the score across for each visualization
    for visualization in np.unique(robustness_data["visualization"]):
        # print(visualization)
        # Since this is on the negative data,
        max_score_for_visualization = np.abs(
            np.min(
                robustness_data[robustness_data.visualization == visualization].result
            )
        )
        # print(max_score_for_visualization)
        robustness_data.loc[
            robustness_data.visualization == visualization, "result"
        ] /= max_score_for_visualization

    complete_dataframe.loc[
        complete_dataframe.perspective == "robustness"
    ] = robustness_data

    return complete_dataframe


def aggregate_metric_for_random(dataset, model, perspective, mean_across_dataset=True):
    if mean_across_dataset:
        data_or_none = aggregate_metric_for_random(
            dataset, model, perspective, mean_across_dataset=False
        )
        if data_or_none is None:
            return None
        else:
            return {k: np.mean(v) for k, v in data_or_none.items()}

    for experiment_index in range(12):
        directory = BASEDIR + f"{dataset}/{model}/experiment_{experiment_index}/"

        if perspective == "faithfulness_ts":
            # Temporal (Un)-Importance
            # filename = f"{dataset}/{model}/experiment_1/temporal_importance/{model}_{dataset}_faithfulness_ti_eval_mean_0.2_auc.h5"
            # data = read_h5(BASEDIR + filename)
            # This dataframe looks like
            #                    result
            #  visualization     {importance: np.array(DATA_POINTS), unimportance: np.array }

            # Temporal Sequence
            filenamerandom = f"temporal_sequence/{model}_{dataset}_faithfulness_ts_eval_mean_0.2_gap_randombase.h5"
            data_random = read_h5(directory + filenamerandom)
            if data_random is None:
                continue
            data_random = data_random.drop(index="random_noabs")
            data_random = data_random.rename(index={"random_abs": "random"})
            # print(data)
            #  Format:           importance              random
            #  visualization     np.array(DATA_POINTS)   np.array(DATA_POINTS)
            # data = data.append(data_random)
            return data_random.to_dict()["importance"]

        if perspective == "faithfulness_ti":
            # Temporal (Un)-Importance
            # filename = f"{dataset}/{model}/experiment_1/temporal_importance/{model}_{dataset}_faithfulness_ti_eval_mean_0.2_auc.h5"
            # data = read_h5(BASEDIR + filename)
            # This dataframe looks like
            #                    methods
            #  visualization     {importance: np.array(DATA_POINTS), unimportance: np.array }

            # Temporal Importance
            filenamerandom = f"temporal_importance/{model}_{dataset}_faithfulness_ti_eval_mean_0.2_predscores_randombase.h5"
            # print(data)
            # return data.to_dict()["importance"]
            scores_random = {}
            ## for random baseline
            data_random = read_h5(directory + filenamerandom)
            if data_random is None:
                continue
            data_random = data_random.drop(index="random_noabs")
            for visualization in data_random.index:
                if visualization == "random":
                    continue
                inner_data = data_random.loc[visualization, "methods"]
                gap_scores = inner_data["importance"]
                if visualization == "random_abs":
                    visualization = "random"
                scores_random[visualization] = gap_scores
            # print(data)
            #  Format:           importance              random
            #  visualization     np.array(DATA_POINTS)   np.array(DATA_POINTS)
            return scores_random

        elif perspective == "sensitivity":
            filenamerandom = f"class_sensitivity/{model}_{dataset}_sensitivity_cs_random_correlation.npy"
            data_random = read_results(directory + filenamerandom)
            if data_random is None:
                continue
            ## put random baseline into data
            # for key in data_random.keys():
            #    data[key] = data_random[key]

            # Data is already in the correct format
            return negate_all(data_random)

        # TODO: This is now robustness.
        elif perspective == "robustness":
            filenamerandom = (
                f"sensitivity_max/{model}_{dataset}_stability_sm_random_scores.npy"
            )
            data_random = read_results(directory + filenamerandom)
            if data_random is None:
                continue
            # data is a dict
            # { pertubation: { visualization : tensor } }
            # We are interested in the 0.2 datapoint for now.
            data_random = data_random["0.2"]
            # for key in data_random.keys():
            #    data[key] = data_random[key]

            # Convert to numpy arrays for consistency.
            data_random = {k: np.array(tensor) for k, tensor in data_random.items()}
            return negate_all(data_random)

        elif perspective == "intraclassstability":
            filenamerandom = f"intraclass_stability/{model}_{dataset}_stability_intraclass_random_dtw.h5"

            ## random baseline
            data_random = read_h5(directory + filenamerandom)
            if data_random is None:
                continue
            results_random = {}
            dtw_random_raw = data_random.loc["dtw_raw"]
            for method in VISUALIZATIONSRANDOM:
                dtw_raw2 = dtw_random_raw[method]
                dtw_raw2 = dtw_raw2["0"]
                upper_triu_dtw1 = np.triu(dtw_raw2)
                upper_triu_dtw2 = np.triu(dtw_raw2)

                ## get only the upper triangle
                upper_triu_dtw1_straight = np.zeros(
                    (
                        int(
                            ((upper_triu_dtw1.shape[0] - 1) * upper_triu_dtw1.shape[1])
                            / 2
                        ),
                        1,
                    )
                )
                upper_triu_dtw2_straight = np.zeros(
                    (
                        int(
                            ((upper_triu_dtw2.shape[0] - 1) * upper_triu_dtw2.shape[1])
                            / 2
                        ),
                        1,
                    )
                )

                ## fill the arrays
                count = 0
                for i in range(upper_triu_dtw1.shape[0]):
                    for j in range(upper_triu_dtw1.shape[1] - i - 1):
                        upper_triu_dtw1_straight[count] = upper_triu_dtw1[i, j + i + 1]
                        # print(i, j+1+i)
                        upper_triu_dtw2_straight[count] = upper_triu_dtw2[i, j + i + 1]
                        count += 1

                # TODO: this should be the mean.
                results_random[method] = float(upper_triu_dtw1_straight.sum())
            return negate_all(results_random)

        elif perspective == "sanity":
            filenamerandom = (
                f"sanity_check/{model}_{dataset}_sanity_Cascade_ssim_random.npy"
            )
            # filename = "sanity_checkFCN_withoutFC_FordA_sanity_Cascade_random_saliencymaps_fcn.0.conv1_abs.npy"

            # Format: {'ssim': layer_id: {visualization: score}}

            ## combine random to data_df
            # data_df = data_df.append(data_random_df)

            data_random = read_results(directory + filenamerandom)
            if data_random is None:
                continue
            # Turn into a visualization x layer dataframe
            data_random_df = pd.DataFrame.from_dict(data_random["ssim"])
            # Take the max across the layers (for now)
            data_random_df["value"] = data_random_df.max(axis=1)
            # Turn the df into a dict, and only use the values for the "value" column to get the right format to return
            return_values_random = data_random_df[["value"]].to_dict()["value"]

            return negate_all(return_values_random)
            # Format: { randomized_layer_id_or_"Original": [accuracy_after_perturbation] }
            # print(data)
            # We need the layer_ids
            # print(data["grads"].shape)

    # Only if we didn't find anything across all experiment runs.
    return None


def aggregate_metric(
    dataset, model, perspective, regularization, regul_para, mean_across_dataset=True
):
    """Reads the necessary files and returns an aggregate metric to concatenate to the dataset.
    Returns a dictionary {visualization: metric}, where metric is an array of results per data point.
    """
    if mean_across_dataset:
        data_or_none = aggregate_metric(
            dataset,
            model,
            perspective,
            regularization,
            regul_para,
            mean_across_dataset=False,
        )
        if data_or_none is None:
            return None
        else:
            return {k: np.mean(v) for k, v in data_or_none.items()}

    # We're running multiple random seeds;
    # for now, this just extracts the first experiment that has data.
    for experiment_index in range(12):
        for perturbation_percent in [0.1, 0.2, 0.5]:
            if regularization is "dropout":
                directory = (
                    BASEDIR
                    + f"{dataset}/{model}/{regularization}_regularization/{regularization}_{regul_para}/experiment_{experiment_index}/"
                )
            elif regularization in ["l1", "l2"]:
                directory = (
                    BASEDIR
                    + f"{dataset}/{model}/{regularization}_regularization/loss_{regul_para}/experiment_{experiment_index}/"
                )
            elif regularization is "no":
                directory = (
                    BASEDIR
                    + f"{dataset}/{model}/no_regularization/no_regularization/experiment_{experiment_index}/"
                )

            if perspective == "faithfulness_ts":
                # Temporal (Un)-Importance
                # filename = f"{dataset}/{model}/experiment_1/temporal_importance/{model}_{dataset}_faithfulness_ti_eval_mean_0.2_auc.h5"
                # data = read_h5(BASEDIR + filename)
                # This dataframe looks like
                #                    result
                #  visualization     {importance: np.array(DATA_POINTS), unimportance: np.array }

                # Temporal Sequence
                if regularization is "dropout":
                    filename = f"temporal_sequence/{model}_{dataset}_{regularization}_regularization_{regul_para}_faithfulness_ts_eval_mean_0.2_gap.h5"
                elif regularization in ["l1", "l2"]:
                    filename = f"temporal_sequence/{model}_{dataset}_{regularization}_regularization_{regul_para}_faithfulness_ts_eval_mean_0.2_gap.h5"
                elif regularization is "no":
                    filename = f"temporal_sequence/{model}_{dataset}_no_regularization_0.0_faithfulness_ts_eval_mean_0.2_gap.h5"

                # filename = f"temporal_sequence/{model}_{dataset}_faithfulness_ts_eval_mean_0.2_gap.h5"
                data = read_h5(directory + filename)

                if data is None:
                    continue
                # print(data)
                #  Format:           importance              random
                #  visualization     np.array(DATA_POINTS)   np.array(DATA_POINTS)
                # data = data.append(data_random)
                return data.to_dict()["importance"]

            if perspective == "faithfulness_ti":
                # Temporal (Un)-Importance
                # filename = f"{dataset}/{model}/experiment_1/temporal_importance/{model}_{dataset}_faithfulness_ti_eval_mean_0.2_auc.h5"
                # data = read_h5(BASEDIR + filename)
                # This dataframe looks like
                #                    methods
                #  visualization     {importance: np.array(DATA_POINTS), unimportance: np.array }

                # Temporal Importance
                filename = f"temporal_importance/{model}_{dataset}_{regularization}_regularization_{regul_para}_faithfulness_ti_eval_mean_{perturbation_percent}_predscores.h5"
                if regularization is "no":
                    filename = f"temporal_importance/{model}_{dataset}_no_regularization_0.0_faithfulness_ti_eval_mean_{perturbation_percent}_predscores.h5"
                data = read_h5(directory + filename)
                if data is None:
                    continue
                # print(data)
                # return data.to_dict()["importance"]

                scores = {}
                for visualization in data.index:
                    if visualization == "random":
                        continue
                    inner_data = data.loc[visualization, "methods"]
                    gap_scores = inner_data[
                        "importance"
                    ]  # - inner_data["unimportance"]
                    scores[visualization] = gap_scores
                # print(data)
                #  Format:           importance              random
                #  visualization     np.array(DATA_POINTS)   np.array(DATA_POINTS)
                return scores

            elif perspective == "sensitivity":
                filename = f"class_sensitivity/{model}_{dataset}_sensitivity_cs_correlation.npy"
                data = read_results(directory + filename)
                if data is None:
                    continue
                ## put random baseline into data
                # for key in data_random.keys():
                #    data[key] = data_random[key]

                # Data is already in the correct format
                return negate_all(data)

            # TODO: This is now robustness.
            elif perspective == "robustness":
                filename = f"sensitivity_max/{model}_{dataset}_stability_sm_scores.npy"
                data = read_results(directory + filename)

                if data is None:
                    continue
                # data is a dict
                # { pertubation: { visualization : tensor } }
                # We are interested in the 0.2 datapoint for now.
                data = data["0.2"]

                # for key in data_random.keys():
                #    data[key] = data_random[key]

                # Convert to numpy arrays for consistency.
                data = {k: np.array(tensor) for k, tensor in data.items()}
                return negate_all(data)

            elif perspective == "intraclassstability":
                filename = f"intraclass_stability/{model}_{dataset}_stability_intraclass_dtw.h5"
                data = read_h5(directory + filename)
                if data is None:
                    continue
                dtw_raw = data.loc["dtw_raw"]

                results = {}
                for method in VISUALIZATIONS:
                    dtw_raw1 = dtw_raw[method]
                    dtw_raw1 = dtw_raw1["0"]
                    upper_triu_dtw1 = np.triu(dtw_raw1)
                    upper_triu_dtw2 = np.triu(dtw_raw1)

                    ## get only the upper triangle
                    upper_triu_dtw1_straight = np.zeros(
                        (
                            int(
                                (
                                    (upper_triu_dtw1.shape[0] - 1)
                                    * upper_triu_dtw1.shape[1]
                                )
                                / 2
                            ),
                            1,
                        )
                    )
                    upper_triu_dtw2_straight = np.zeros(
                        (
                            int(
                                (
                                    (upper_triu_dtw2.shape[0] - 1)
                                    * upper_triu_dtw2.shape[1]
                                )
                                / 2
                            ),
                            1,
                        )
                    )

                    ## fill the arrays
                    count = 0
                    for i in range(upper_triu_dtw1.shape[0]):
                        for j in range(upper_triu_dtw1.shape[1] - i - 1):
                            upper_triu_dtw1_straight[count] = upper_triu_dtw1[
                                i, j + i + 1
                            ]
                            # print(i, j+1+i)
                            upper_triu_dtw2_straight[count] = upper_triu_dtw2[
                                i, j + i + 1
                            ]
                            count += 1

                    # TODO: this should be the mean.
                    results[method] = float(upper_triu_dtw1_straight.sum())

                return negate_all(results)

            elif perspective == "sanity":
                filename = (
                    f"sanity_check/{model}_{dataset}_sanity_Cascade_ssim_final.npy"
                )
                # filename = "sanity_checkFCN_withoutFC_FordA_sanity_Cascade_random_saliencymaps_fcn.0.conv1_abs.npy"
                data = read_results(directory + filename)
                if data is None:
                    continue
                # Format: {'ssim': layer_id: {visualization: score}}

                # Turn into a visualization x layer dataframe
                data_df = pd.DataFrame.from_dict(data["ssim"])
                ## combine random to data_df
                # data_df = data_df.append(data_random_df)

                # Take the max across the layers (for now)
                data_df["value"] = data_df.max(axis=1)

                # Turn the df into a dict, and only use the values for the "value" column to get the right format to return
                return_values = data_df[["value"]].to_dict()["value"]
                return negate_all(return_values)
                # Format: { randomized_layer_id_or_"Original": [accuracy_after_perturbation] }
                # print(data)
                # We need the layer_ids
                # print(data["grads"].shape)
            elif perspective == "accuracy":
                filename = f"best_model.csv"
                try:
                    data = pd.read_csv(directory + filename)
                    print(data)

                    return {vis: data["test_accuracy"][0] for vis in VISUALIZATIONS}
                except FileNotFoundError:
                    continue
    # Only if we didn't find anything across all experiment runs.
    return None


results_dataframe = None


def read_data():
    global results_dataframe
    # global results_dataframe_random

    data = []
    # data_random = [] ## for the random baseline
    for dataset, model, perspective, regularization, regul_para in itertools.product(
        DATASETS, MODELS, PERSPECTIVES, REGULARIZATIONS, REGUL_PARAS
    ):
        results = aggregate_metric(
            dataset, model, perspective, regularization, regul_para
        )
        if results is not None:
            # Mean dataset
            # means = { k: result.mean() for k, result in results.items() }
            # means["perspective"] = perspective
            # means["model"] = model
            # means["dataset"] = dataset
            # data.append(means)

            results_dictionaries = [
                dict(
                    dataset=dataset,
                    perspective=perspective,
                    model=model,
                    regularization=regularization,
                    regu_parameter=regul_para,
                    visualization=visualization,
                    result=r,
                )
                for visualization, values in results.items()
                for r in (
                    values if isinstance(values, (list, np.ndarray)) else [values]
                )
            ]
            data.extend(results_dictionaries)
        else:
            print(
                f"[Normal Data] Missing Data for {dataset}/{model}/{perspective}/{regularization}_regularization/{regul_para}."
            )

    ## random baseline
    # for dataset, model, perspective in itertools.product(
    #     DATASETS, MODELS, PERSPECTIVES
    # ):
    #     results_random = aggregate_metric_for_random(dataset, model, perspective)
    #     if results_random is not None:
    #         print(results_random)
    #     else:
    #         print(
    #             f"[Random Baseline] Missing Data for {dataset}/{model}/{perspective}."
    #         )

    # Mean dataset
    # means = { k: result.mean() for k, result in results.items() }
    # means["perspective"] = perspective
    # means["model"] = model
    # means["dataset"] = dataset
    # data.append(means)

    ## random baseline dictionaries
    #        results_dictionaries_random = [
    #            dict(
    #                dataset=dataset,
    #                perspective=perspective,
    #                model=model,
    #                visualization=visualization,
    #                result=r,
    #            )
    #            for visualization, values in results_random.items()
    #            for r in (values if isinstance(values, (list, np.ndarray)) else [values])
    #        ]

    #        data_random.extend(results_dictionaries_random)
    #    else:
    #        print(f"[Random Baseline] Missing Data for {dataset}/{model}/{perspective}.")

    dataframe = pd.DataFrame(
        columns=["dataset", "perspective", "model", "visualization", "result"]
    )
    dataframe = dataframe.append(data)
    ## for random baseline
    # dataframe_random = pd.DataFrame(columns=["dataset", "perspective", "model", "visualization", "result"])
    # dataframe_random = dataframe_random.append(data_random)

    # dataframe = normalize_robustness(dataframe)
    # dataframe_random = normalize_robustness(dataframe_random)
    # print(dataframe)

    print("Data read:")
    print(dataframe)

    # print("Data random baseline read:")
    # print(dataframe_random)

    results_dataframe = dataframe
    # results_dataframe_random = dataframe_random


# print(data)
read_data()

## Additional Notes
# Faithfulness:
#   temporal_importance/importance? 0.2! percent _ auc (mean / zero)
#   temporal_sequence
# sensitivity_check = class_sensitivity
# stability_check = (/ intra-class-stability?) sensitivity_max
# localization = f1 classic
# sanity_check = cascade / independent?
#       mean ssem(layers_cascade)
#       -max ssem(layers)
# iosr ==> recall/precision für back/flat/front/middle

# 21_08_timeseries_interpretability/results/tooltracking/FCN_withoutFC/class_sensitivity/experiment_14_Tool_fcn_pool_class_sensitivity_correlation.npy
# results = read_results("tooltracking/FCN_withoutFC/class_sensitivity/experiment_14_Tool_fcn_pool_class_sensitivity_correlation.npy") #experiment_14_class_sensitivity_max_explanation.npy")

# results = read_results("tooltracking/FCN_withoutFC/sensitivity_check/experiment_14_Tool_fcn_pool_sensitivity_max_scores010.npy")

# print(results.keys())

# mean = np.mean(results["integrated_gradients"].numpy())
# print(mean)
# take mean and variance across dataset

# temporal importance
# 0.2 percent is threshold (20%) bis zu der perturbiert

# Correct results are sample-wise?
# ==> need to aggregate them.
