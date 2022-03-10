# %%

"Plot regularization vs. Faithfulness"

import random
from numpy.lib.histograms import _ravel_and_check_weights
import seaborn as sns
import pandas as pd
# from paperplots import results_reader as results
import matplotlib.pyplot as plt
# from paperplots.analysis import results_dataframe, results_dataframe_without_datasets, normalize_range_per_perspective, results_dataframe_without_datasets_random
# from paperplots.analysis import normalize_range_random
from plot_radar import plot_radar_chart, convert_to_radar_dataframe
import results_reader_regularization

# %%

replacements = {
    "faithfulness_ti": "Faithfulness TI",
    "faithfulness_ts": "Faithfulness TS",
    "sensitivity": "Sensitivity",
    "robustness": "Robustness",
    "sanity": "Sanity",
    "intraclassstability": "Intra-Class Stability",

    "FCN_withoutFC": "FCN",
    "TCN_withoutFC": "TCN",

    "dropout": "Dropout Regularization",
    "l1": "L1 Regularization",
    "l2": "L2 Regularization",
    "None": "No Regularization",

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
    "random_abs": "Random Baseline"
}


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
        order = [replacements.get(p) for p in results_reader_regularization.PERSPECTIVES if
                 replacements.get(p) in dataframe.columns]
        order += ["name", "dtype"]
    else:
        order = [replacements.get(p) for p in results_reader_regularization.PERSPECTIVES if
                 replacements.get(p) in dataframe.columns]
    # print(order)
    return_value = dataframe[order]
    # print(return_value)
    return return_value


FIG_3_SCALE = 0.85

## read results
results_reader_regularization.read_data()

dataframe = results_reader_regularization.results_dataframe.copy()
df = prettify_dataframe(dataframe)
print("Unique perspectives:", df.perspective.unique())


# df[df.dataset == "FordA"][df.perspective == "Faithfulness TI"][df.regularization == "Dropout Regularization"]

# %%


def plot_regularization_trend(ax, df, regularization="dropout", dataset="FordA", perspective="Faithfulness TI"):
    # fig, ax = plt.subplots(1, 1, figsize=(6,4))

    visualizations = list(df.visualization.unique())
    for visualization in visualizations:
        df_filter = df[df.dataset == dataset]
        df_filter = df_filter[df_filter.perspective == perspective]
        df_filter = df_filter[df_filter.regularization == regularization]
        df_filter = df_filter[df_filter.visualization == visualization]
        df_filter.plot(x="regu_parameter", y="result", ax=ax, label=visualization)
    ax.legend(loc='lower left')
    ax.set_title(dataset)
    ax.set_xlabel(regularization)
    ax.set_ylabel(perspective)
    df_filter = df[df.dataset == dataset]
    df_filter = df_filter[df_filter.perspective == perspective]
    df_filter = df_filter[df_filter.regularization == regularization]
    df_filter = df_filter[df_filter.visualization == visualization]
    ticks = list(df_filter.regu_parameter.unique())
    ax.set_xticks(ticks=ticks)


fig, axs = plt.subplots(2, 3, figsize=((16, 10)))
# fig.tight_layout()
dataset = "FordA"
# plot_regularization_trend(ax=axs[0, 0], df=df, regularization="No Regularization", perspective="Faithfulness TI", dataset=dataset)
plot_regularization_trend(ax=axs[0, 0], df=df, regularization="Dropout Regularization", perspective="Faithfulness TI",
                          dataset=dataset)
plot_regularization_trend(ax=axs[0, 1], df=df, regularization="L1 Regularization", perspective="Faithfulness TI",
                          dataset=dataset)
plot_regularization_trend(ax=axs[0, 2], df=df, regularization="L2 Regularization", perspective="Faithfulness TI",
                          dataset=dataset)

# plot_regularization_trend(ax=axs[1, 0], df=df, regularization="No Regularization", perspective="Faithfulness TS", dataset=dataset)
plot_regularization_trend(ax=axs[1, 0], df=df, regularization="Dropout Regularization", perspective="Faithfulness TS",
                          dataset=dataset)
plot_regularization_trend(ax=axs[1, 1], df=df, regularization="L1 Regularization", perspective="Faithfulness TS",
                          dataset=dataset)
plot_regularization_trend(ax=axs[1, 2], df=df, regularization="L2 Regularization", perspective="Faithfulness TS",
                          dataset=dataset)
plt.show()

# %%

import numpy as np

print(df[df.regularization == "No Regularization"][df.regu_parameter == 0.200][df.dataset == 'FordB'][
          df.perspective == "Faithfulness TI"].groupby("dataset").agg([np.mean, np.std]))
print(df[df.regularization == "No Regularization"][df.regu_parameter == 0.200][df.dataset == 'FordB'][
          df.perspective == "Faithfulness TS"].groupby("dataset").agg([np.mean, np.std]))

dataset = 'FordB'
perspective = "Faithfulness TS"

df_temp = df[df.regularization == "No Regularization"][df.perspective == perspective][df.regu_parameter == 0.200][
    df.dataset == dataset]

df_temp.groupby("dataset").agg([np.mean, np.std]).result

# %%

import numpy as np

fig, axs = plt.subplots(1, 3, figsize=((12, 3)))
regularizations = ["Dropout Regularization", "L1 Regularization", "L2 Regularization"]
perspective = "Faithfulness TI"
for i in range(0, 3):
    ax = axs[i]
    regularization = regularizations[i]
    datasets = ["FordA", "FordB", "NATOPS"]
    for dataset in datasets:
        df_filter = df[df.dataset == dataset]
        df_filter = df_filter[df_filter.perspective == perspective]
        df_filter = df_filter[df_filter.regularization == regularization]
        regu_parameter = df_filter.groupby("regu_parameter").agg([np.mean, np.std])
        a, b = regu_parameter
        regu_parameter.plot(y=a, ax=ax, label=dataset, kind='line', linestyle='--', marker='x', markersize=10)

        # baseline
        # print(df[df.regularization == "No Regularization"][df.regu_parameter == 0.200])
        df_baseline = \
        df[df.regularization == "No Regularization"][df.perspective == perspective][df.regu_parameter == 0.200][
            df.dataset == dataset]
        # print(df_baseline)
        df_baseline = df_baseline.groupby("dataset").agg([np.mean, np.std]).result
        # print(df_baseline)

        a, b = df_baseline
        print(df_baseline)
        #df_baseline.plot(y=a, ax=ax, label=dataset, kind='line', linestyle='--', marker='o', markersize=10)
        mean_baseline = list(df_baseline.mean())[0]
        yerr_baseline = list(df_baseline.mean())[1]

        # print(df_baseline)
        # [df_filter.perspective == perspective]
        # df_baseline.plot(y=a, ax=ax, label=dataset, kind='line', linestyle='--', marker='o', markersize=10)

        ax.set_ylim(0.0, 0.7)
        # ax.set_ylim(0.4, 0.7)
        ax.legend(loc='lower left')
        if regularization == "Dropout Regularization":
            ax.set_ylabel(perspective)
            ax.set_xlim(0.0, 0.7)
            ax.set_xticks([0.2, 0.5])
            ax.errorbar(y=mean_baseline, yerr=yerr_baseline, x=0.2)
            ax.errorbar(y=mean_baseline, yerr=yerr_baseline, x=0.5)
        if regularization == "L1 Regularization":
            ax.legend(loc='upper left')
            ax.set_xlim(-0.005, 0.015)
            ax.set_xticks([0.001, 0.01])
        if regularization == "L2 Regularization":
            ax.set_xlim(-0.005, 0.015)
            ax.set_xticks([0.001, 0.01])
        ax.set_title(regularization)
        ax.set_xlabel(regularization + " parameters")

filename = "reg_faith_ti.pdf"
plt.savefig(filename,
            format="pdf",
            dpi=None,
            facecolor='w',
            edgecolor='k',
            orientation='portrait',
            papertype=None,
            transparent=False,
            bbox_inches='tight',
            pad_inches=0.0,
            frameon=None,
            metadata=None)
plt.show()

# %%

fig, axs = plt.subplots(1, 3, figsize=((12, 3)))
regularizations = ["Dropout Regularization", "L1 Regularization", "L2 Regularization"]
perspective = "Faithfulness TS"
for i in range(0, 3):
    ax = axs[i]
    regularization = regularizations[i]
    datasets = ["FordA", "FordB", "NATOPS"]
    for dataset in datasets:
        df_filter = df[df.dataset == dataset]
        df_filter = df_filter[df_filter.perspective == perspective]
        df_filter = df_filter[df_filter.regularization == regularization]
        regu_parameter = df_filter.groupby("regu_parameter").agg([np.mean, np.std])
        a, b = regu_parameter
        regu_parameter.plot(y=a, ax=ax, label=dataset, kind='line', linestyle='--', marker='x', markersize=10)
        ax.set_ylim(0.0, 0.32)
        ax.legend(loc='lower left')
        if regularization == "Dropout Regularization":
            ax.set_ylabel(perspective)
            ax.legend(loc='upper right')
            ax.set_xlim(0.0, 0.7)
            ax.set_xticks([0.2, 0.5])
        if regularization == "L1 Regularization":
            ax.legend(loc='upper right')
            ax.set_xlim(-0.005, 0.015)
            ax.set_xticks([0.001, 0.01])
        if regularization == "L2 Regularization":
            ax.legend(loc='center left')
            ax.set_xlim(-0.005, 0.015)
            ax.set_xticks([0.001, 0.01])
        ax.set_title(regularization)
        ax.set_xlabel(regularization + " parameters")
filename = "reg_faith_ts.pdf"
plt.savefig(filename,
            format="pdf",
            dpi=None,
            facecolor='w',
            edgecolor='k',
            orientation='portrait',
            papertype=None,
            transparent=False,
            bbox_inches='tight',
            pad_inches=0.0,
            frameon=None,
            metadata=None)
plt.show()

# %%


