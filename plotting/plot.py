import random
from numpy.lib.histograms import _ravel_and_check_weights
import seaborn as sns
import pandas as pd
import results_reader as results
import matplotlib.pyplot as plt
from analysis import (
    results_dataframe,
    results_dataframe_without_datasets,
    normalize_range_per_perspective,
    results_dataframe_without_datasets_random,
)
from analysis import normalize_range_random
from plot_radar import plot_radar_chart, convert_to_radar_dataframe

import results_reader

replacements = {
    "faithfulness_ti": "Faithfulness TI",
    "faithfulness_ts": "Faithfulness TS",
    "sensitivity": "Sensitivity",
    "robustness": "Robustness",
    "sanity": "Sanity",
    "intraclassstability": "Intra-Class Stability",
    "FCN_withoutFC": "FCN",
    "TCN_withoutFC": "TCN",
    "grads": "Gradient",
    "smoothgrads": "SmoothGrads",
    "igs": "Int. Gradients",
    "lrp_epsilon": "LRP",
    "gradCAM": "GradCAM",
    "guided_gradcam": "Guided GradCAM",
    "guided_backprop": "Guided Backprop.",
    "lime": "LIME",
    "kernel_shap": "Kernel-SHAP",
    "random": "Random Baseline",
    "random_abs": "Random Baseline",
}

FIG_3_SCALE = 0.85


def prettify_dataframe(dataframe, columns=False, rows=False):
    """Changes labels into their paper equivalents."""
    if rows:
        return dataframe.rename(index=replacements).replace(replacements)

    if columns:
        return dataframe.rename(columns=replacements).replace(replacements)
    else:
        return dataframe.replace(replacements)


def reorder_heatmap_dataframe(dataframe):
    # print(dataframe)
    # print(dataframe.columns)
    if "name" in dataframe.columns:
        order = [
            replacements.get(p)
            for p in results_reader.PERSPECTIVES
            if replacements.get(p) in dataframe.columns
        ]
        order += ["name", "dtype"]
    else:
        order = [
            replacements.get(p)
            for p in results_reader.PERSPECTIVES
            if replacements.get(p) in dataframe.columns
        ]
    # print(order)
    return_value = dataframe[order]
    # print(return_value)
    return return_value


def plot_results():
    overview_dataframe = prettify_dataframe(results_dataframe.copy())
    print(overview_dataframe)
    plot = sns.catplot(
        y="visualization",
        x="result",
        col="perspective",
        # row="model",
        # hue="dataset",
        data=overview_dataframe,
        kind="box",
        sharex="col",
        margin_titles=True,
        aspect=0.6 * 1.5,
        height=6 / 1.5 * FIG_3_SCALE,
    )  # , style="dataset"
    (
        plot.set_axis_labels("", "").set_titles(
            col_template="{col_name}", row_template="{row_name}"
        )  # {col_var}
        # .set(xlim=(-0.05, 1.05))
        .despine(top=False, right=False)
        # .set_xticks([0, 0.5, 1])
        # .set_xticklabels(["0", "0.5", "1"])
    )
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.gcf().savefig("output/overview_results_random.pdf")
    # plot.figure
    # plot.figure


def plot_results_swarm():
    overview_dataframe = prettify_dataframe(results_dataframe.copy())
    plot = sns.catplot(
        y="visualization",
        x="result",
        col="perspective",
        # row="model",
        hue="dataset",
        data=overview_dataframe,
        kind="swarm",
        sharex="col",
        margin_titles=True,
        aspect=0.9 * 1.5,
        height=4 / 1.5,
    )  # , style="dataset"
    (
        plot.set_axis_labels("", "")
        .set_titles(col_template="{col_name}", row_template="{row_name}")  # {col_var}
        .set(xlim=(-0.05, 1.05))
        .despine(top=False, right=False)
        # .set_xticks([0, 0.5, 1])
        # .set_xticklabels(["0", "0.5", "1"])
    )
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.gcf().savefig("output/overview_results_swarm_radom.pdf")


def plot_dataset_influence():
    # results_dataframe = prettify_dataframe(results_dataframe)
    overview_dataframe = prettify_dataframe(results_dataframe.copy())
    plot = sns.catplot(
        y="dataset",
        x="result",
        col="perspective",
        # hue="dataset",
        data=overview_dataframe,
        kind="box",
        sharex="col",
        margin_titles=True,
        aspect=0.9 * 1.5 / 0.7,
        height=4
        / 1.5
        * (0.75)
        * FIG_3_SCALE,  # So that this plot has approx. same "box-height" as the big results plot.
    )  # , style="dataset"
    (
        plot.set_axis_labels("", "")
        .set_titles(col_template="{col_name}", row_template="{row_name}")  # {col_var}
        .set(xlim=(-0.05, 1.05))
        .despine(top=False, right=False)
        # .set_xticks([0, 0.5, 1])
        # .set_xticklabels(["0", "0.5", "1"])
    )
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.gcf().savefig("output/overview_datasetinfluence.pdf")


def plot_results_without_datasets():
    # results_dataframe = prettify_dataframe(results_dataframe)
    overview_dataframe = prettify_dataframe(results_dataframe_without_datasets.copy())
    plot = sns.catplot(
        y="visualization",
        x="result",
        col="perspective",
        # hue="dataset",
        data=overview_dataframe,
        kind="box",
        sharex="col",
        margin_titles=True,
        aspect=0.9,
        height=4,
    )  # , style="dataset"
    (
        plot.set_axis_labels("", "").set_titles(
            col_template="{col_name}", row_template="{row_name}"
        )  # {col_var}
        # .set(xlim=(-1.5, 1.5))
        .despine(top=False, right=False)
        # .set_xticks([0, 0.5, 1])
        # .set_xticklabels(["0", "0.5", "1"])
    )
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.gcf().savefig("output/overview_withoutdatasets_random.pdf")


def plot_results_models():
    # results_dataframe = prettify_dataframe(results_dataframe)
    overview_dataframe = prettify_dataframe(results_dataframe.copy())
    plot = sns.catplot(
        y="model",
        x="result",
        col="perspective",
        # hue="dataset",
        data=overview_dataframe,
        kind="box",
        sharex="col",
        margin_titles=True,
        aspect=2,
        height=3.6 / 2,  # Can't use width here, since that messes up the box plot.
    )  # , style="dataset"
    (
        plot.set_axis_labels("", "")
        .set_titles(col_template="{col_name}", row_template="{row_name}")  # {col_var}
        .set(xlim=(-0.05, 1.05))
        .despine(top=False, right=False)
        # .set_xticks([0, 0.5, 1])
        # .set_xticklabels(["0", "0.5", "1"])
    )
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.gcf().savefig("output/overview_models.pdf")


# plot_results()
# plot_dataset_influence()
# plot_results_without_datasets()
# plot_results_models()
# plot_results_swarm()


# fig.suptitle('Deployment LeNet, MNIST', fontsize='large', y=1.0)

# print(results_dataframe)


def plot_radar(results, name):
    df = results
    # df = data[data["visualization"] == col]
    radar_df = convert_to_radar_dataframe(
        prettify_dataframe(df), "visualization", "perspective"
    )
    radar_df = reorder_heatmap_dataframe(radar_df)
    # We need to do this if some data is missing.
    perspectives_for_chart = [
        col for col in radar_df.columns if not col in ["name", "dtype"]
    ]
    # Re-sort the columns here for consistency.
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection="polar"))
    h1, l1 = plot_radar_chart(
        radar_df, perspectives_for_chart, f"", ax, min=-0.05, max=1.05, fix_cols=True
    )
    fig.tight_layout()
    ax.legend(h1, l1, loc="upper right", bbox_to_anchor=(1.1, 1.1))  # , ncol=2
    fig.savefig(f".//output/{name}")
    plt.close()


def convert_to_heatmap_dataframe(tidy_df, layers, axes):
    """Converts a standard-format tidy dataframe to the format needed
    by the radar-plot function.

    layers: Which column to use as visual layers in the dataframe.
    axes: Which column to use as the axes of the radar chart.
    """
    df = tidy_df.copy()
    radar_df_dict = df.pivot_table(index=[layers], columns=axes)
    radar_df_dict.columns = radar_df_dict.columns.droplevel().rename(None)
    return radar_df_dict


def plot_heatmap(results, name, row, col):
    # Unpivot the dataframe
    results = prettify_dataframe(results)
    results = convert_to_heatmap_dataframe(results, row, col)
    results = reorder_heatmap_dataframe(results).T
    fig, ax = plt.subplots(1, 1)
    ax = sns.heatmap(
        results,
        square=True,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cmap="magma",
        annot=True,
        cbar=False,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    fig.tight_layout()
    fig.savefig(f".//output/{name}")
    plt.close()


def plot_radar_for_model_dataset_and_methods(
    dataframe, model, dataset, methods, normalize=False
):
    data = dataframe[dataframe.model == model]
    data = data[data.dataset == dataset]
    if methods is not None:
        data = data[[m in methods for m in data.visualization]]
    plot_heatmap(
        data,
        f"heatmap_{model}_{dataset}_{methods}_random.pdf",
        "visualization",
        "perspective",
    )
    if normalize:
        data = normalize_range_per_perspective(data)
    plot_radar(data, f"radar_{model}_{dataset}_{methods}_random.pdf")


(
    radar_plot_base_data,
    radar_plot_base_data_random,
    perspective_minmax,
) = normalize_range_per_perspective(
    results_dataframe_without_datasets, results_dataframe_without_datasets_random
)
radar_plot_base_data = radar_plot_base_data.append(radar_plot_base_data_random)
# results_dataframe_without_datasets_random = normalize_range_random(results_dataframe_without_datasets_random, perspective_minmax)
# radar_plot_base_data = radar_plot_base_data.append(results_dataframe_without_datasets_random)
print(radar_plot_base_data)
# plot_radar(radar_plot_base_data[radar_plot_base_data["model"] == "TCN_withoutFC"], "radar_TCN.pdf")
# plot_radar_for_model_dataset_and_methods(radar_plot_base_data, "TCN_withoutFC", "FordB", ["lrp_epsilon", "kernel_shap", "gradCAM", "grads"])

# plot_radar_for_model_dataset_and_methods(radar_plot_base_data, "TCN_withoutFC", "FordA", ["grads", "lime", "kernel_shap"])

# Plots that Chris and Lukas decided on
# plot_radar_for_model_dataset_and_methods(radar_plot_base_data, "TCN_withoutFC", "FordA", ["grads", "igs", "kernel_shap"])
# plot_radar_for_model_dataset_and_methods(radar_plot_base_data, "TCN_withoutFC", "FordB", ["grads", "guided_gradcam", "kernel_shap"])
# plot_radar_for_model_dataset_and_methods(radar_plot_base_data, "TCN_withoutFC", "GunPointAgeSpan", ["grads", "gradCAM", "smoothgrads"])
# plot_radar_for_model_dataset_and_methods(radar_plot_base_data, "TCN_withoutFC", "NATOPS", ["grads", "igs", "kernel_shap"]) # guided gradcam


def plot_correlations(dataframe, postfix):
    # Scores for one metric should not be correlated to scores from other metrics
    dataframe = prettify_dataframe(dataframe.copy())
    dataframe = dataframe[dataframe.perspective != "Faithfulness TS"]

    long_form_df = dataframe.pivot_table(
        index=["dataset", "model", "visualization"], columns="perspective"
    )
    long_form_df = prettify_dataframe(long_form_df, columns=True)
    print(long_form_df)
    correlations = long_form_df.corr()
    print(correlations)
    correlations.columns = correlations.columns.droplevel().rename(None)
    correlations.index = correlations.index.droplevel().rename(None)

    fig, ax = plt.subplots(1, 1)
    ax = sns.heatmap(
        correlations,
        square=True,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cmap="magma",
        annot=True,
        cbar=False,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(f"output/metric_correlations_{postfix}.pdf")
    plt.close()

    long_form_df.columns = long_form_df.columns.droplevel().rename(None)
    long_form_df.reset_index(inplace=True)
    long_form_df = prettify_dataframe(long_form_df)
    # print(long_form_df)
    fig = sns.pairplot(
        long_form_df,
        hue="visualization",
        markers=[
            ",",
            ".",
            "o",
            "v",
            "^",
            "<",
            ">",
            "8",
            "s",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
        ][:9],
    )
    fig.tight_layout()
    fig.savefig(f"output/metric_pairplots_{postfix}.pdf")
    plt.close()


# plot_correlations(results_dataframe, "normal")
# plot_correlations(results_dataframe_without_datasets, "without_datasets")

for dataset in set(results_dataframe_without_datasets.dataset):
    plot_radar_for_model_dataset_and_methods(
        radar_plot_base_data, "TCN_withoutFC", dataset, None
    )