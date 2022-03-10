"Plot regularization vs. Faithfulness"

import random
from numpy.lib.histograms import _ravel_and_check_weights
import seaborn as sns
import pandas as pd
#from paperplots import results_reader as results
import matplotlib.pyplot as plt
#from paperplots.analysis import results_dataframe, results_dataframe_without_datasets, normalize_range_per_perspective, results_dataframe_without_datasets_random
#from paperplots.analysis import normalize_range_random
from paperplots.plot_radar import plot_radar_chart, convert_to_radar_dataframe
from paperplots import results_reader_regularization

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
    "random_abs": "Random Baseline"
}

FIG_3_SCALE = 0.85

## read results
results_reader_regularization.read_data()

dataframe = results_reader_regularization.results_dataframe.copy()

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
        order = [ replacements.get(p) for p in results_reader_regularization.PERSPECTIVES if replacements.get(p) in dataframe.columns]
        order += ["name", "dtype"]
    else:
        order = [ replacements.get(p) for p in results_reader_regularization.PERSPECTIVES if replacements.get(p) in dataframe.columns]
    # print(order)
    return_value = dataframe[order]
    # print(return_value)
    return return_value

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
    ax = sns.heatmap(results, square=True, linewidths=0.5, vmin=0, vmax=1, cmap="magma", annot=True, cbar=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    fig.tight_layout()
    fig.savefig(f"../plots_regularized/output/{name}")
    plt.close()
    
def plot_radar_for_model_dataset_and_methods(dataframe, model, dataset, methods, normalize=False):
    data = dataframe[dataframe.model == model]
    data = data[data.dataset == dataset]
    if methods is not None:
        data = data[[m in methods for m in data.visualization]]
    plot_heatmap(data, f"heatmap_{model}_{dataset}_{methods}_random.pdf", "visualization", "perspective")

print(dataframe)
plot_radar_for_model_dataset_and_methods(dataframe=dataframe,
                                         model="FCN_withoutFC",
                                         dataset="FordA",
                                         methods=None#"smoothgrads"
                                         )