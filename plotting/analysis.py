"""Converts and analyses the results."""

import results_reader
import pandas as pd
import numpy as np


results_reader.read_data()

dataframe = results_reader.results_dataframe.copy()
## random baseline
dataframe_random = results_reader.results_dataframe_random.copy()


# Remove the influence of the dataset
# by substracting the per-perspective and per-dataset mean from the scores.

for perspective in set(dataframe.perspective):
    for dataset in set(dataframe.dataset):
        mask = (dataframe.perspective == perspective) & (dataframe.dataset == dataset)

        dataframe.loc[mask, "result"] -= np.mean(dataframe[mask].result)
        dataframe.loc[mask, "result"] /= np.std(dataframe[mask].result)

        ## for random baseline
        dataframe_random.loc[mask, "result"] -= np.mean(dataframe[mask].result)
        dataframe_random.loc[mask, "result"] /= np.std(dataframe[mask].result)


# Perspective-Wise Data Normalization
# Normalize data s.t. the 99th percentile is 1, and the first percentile is 0
def normalize_range_per_perspective(dataframe, dataframe_random):
    dataframe = dataframe.copy()
    dataframe_random = dataframe_random.copy()

    perspective_minmax_dataframe = pd.DataFrame(columns=["perspective", "maxi", "mini"])
    perspective_minmax_list = []

    for perspective in results_reader.PERSPECTIVES:
        data = dataframe[dataframe.perspective == perspective]["result"]
        data_random = dataframe_random[dataframe_random.perspective == perspective][
            "result"
        ]
        # print(data)
        maximum = np.max(data)  # np.nanpercentile(data, 99)
        minimum = np.min(data)  # np.nanpercentile(data, 1)

        # print(f"Perspective {perspective}: Max={maximum}, min={minimum}")
        dataframe.loc[dataframe.perspective == perspective, "result"] = (
            data - minimum
        ) / (maximum - minimum)
        dataframe_random.loc[dataframe_random.perspective == perspective, "result"] = (
            data_random - minimum
        ) / (maximum - minimum)

        result_minmax = [dict(perspective=perspective, maxi=maximum, mini=minimum)]
        perspective_minmax_list.extend(result_minmax)
    perspective_minmax_dataframe.append(perspective_minmax_list)
    return dataframe, dataframe_random, perspective_minmax_dataframe


def normalize_range_random(dataframe, perspective_minmax_dataframe):
    ## normlize random baseline
    dataframe1 = dataframe.copy()
    for perspective in results_reader.PERSPECTIVES:
        data = dataframe1[dataframe1.perspective == perspective]["result"]

        maximum = perspective_minmax_dataframe[dataframe1.perspective == perspective][
            "maxi"
        ]  # np.nanpercentile(data, 99)
        minimum = perspective_minmax_dataframe[dataframe1.perspective == perspective][
            "mini"
        ]  # np.nanpercentile(data, 1)

        # print(f"Perspective {perspective}: Max={maximum}, min={minimum}")
        dataframe1.loc[dataframe1.perspective == perspective, "result"] = (
            data - minimum
        ) / (maximum - minimum)

    return dataframe1


# results_dataframe = results_reader.dataframe
(
    results_dataframe,
    results_dataframe_random,
    perspective_minmax_dataframe,
) = normalize_range_per_perspective(
    results_reader.results_dataframe, results_reader.results_dataframe_random
)
## add random baseline
# dataframe = dataframe.append(dataframe_random)
results_dataframe_without_datasets = (
    dataframe  # normalize_range_per_perspective(dataframe)
)
results_dataframe_without_datasets_random = dataframe_random

## random baseline
results_dataframe_random = normalize_range_random(
    results_reader.results_dataframe_random, perspective_minmax_dataframe
)
results_dataframe = results_dataframe.append(results_dataframe_random)

results_dataframe.to_csv("./output/results_dataframe_random.csv")
results_dataframe_without_datasets.to_csv(
    "./output/results_dataframe_without_datasets_with_random.csv"
)
# print(results_dataframe_without_datasets)
