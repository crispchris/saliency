"""
load and view dataset
For Dataset UCR and Dataset Tool Tracking

"""

## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import argparse
import numpy as np
import matplotlib.pyplot as plt

## ---------------------
## --- Tool-Tracking ---
## ---------------------
from datatools import ACC, GYR, MAG, MIC, POS, VEL

## -----------
## --- Own ---
## -----------
from utils import plot_dataset, read_dataset_ts, plot_sample_per_dimension, plot_samples_each_class
from utils import create_directory
from tool_tracking.dataknowing.loadData import load_data, pipe_segment, filteroutlabels, balance_classes_oneinput
from tool_tracking.dataknowing.loadData import summarize_labels, balance_classes, split_data, split_data_oneinput
from tool_tracking.dataknowing.loadData import merge_labels, filteroutlabels_one_array, zero_normalization
from tool_tracking.tool_utils import Traindata



# if __name__ == '__main__':
## Arguments
parser = argparse.ArgumentParser(description="Datasets View")
parser.add_argument("--dataset", type=str, default="tool_tracking", help="the name of Dataset")
parser.add_argument("--tool", type=str, default="electric_screwdriver", help="the tool name if tool_tracking is used")
parser.add_argument("--window", type=bool, default=True, help="whether window the samples if tool_tracking or stabilo is used")
parser.add_argument("--window_len", type=float, default=0.2, help="window length if window is used in tool_tracking or stabilo is used")
parser.add_argument("--window_overlap", type=float, default=0.5, help="window overlap if window is used in tool_tracking or stabilo is used")
parser.add_argument("--plot_dimension", type=bool, default=False, help="Plot classes per dimension")
parser.add_argument("--plot_one_sample", type=bool, default=True, help="Plot a sample per class")
parser.add_argument("--plot_samples", type=bool, default=False, help="Plot some samples in one plot per class")

args = parser.parse_args()
dataset_name = args.dataset
dataset_name = "GunPointAgeSpan"
# Load the dataset
if dataset_name is "tool_tracking":
    ## here only 4 classes will be viewed
    ## classes = {2, 3, 4, 5}
    source = "data/tool-tracking/tool-tracking-data"
    tool = args.tool
    sensors = ['acc', 'gyr', 'mag']
    sensors_list = [ACC, GYR, MAG]
    garbage_labels = [-1, 6, 7, 8, 14]
    label_summary = {0:[2], 1:[3], 2:[4], 3:[5]}
    ## set window length and overlap
    if args.window:
        window_length = args.window_len
        window_overlap = args.window_overlap

    ## Load Data
    data_dict = load_data(source=source, tool=tool, datatypes=sensors_list) ## Each measurement is a DataBunch
    ## Windowing, into one array
    dataset, yt = pipe_segment(data=data_dict, window_length=window_length, overlap=window_overlap,
                               enforce_size=True, sensors=sensors, discard_time=True,
                               one_array=True)

    ## Filter out garbage labels
    dataset, yt = filteroutlabels_one_array(dataset, labels=yt, garbage_labels=garbage_labels)
    reYt = merge_labels(yt, label_mapping=label_summary, print_info=True)

    ## zero mean normalization
    data = None
    for i in range(dataset.shape[0]):
        if data is None:
            data = dataset[i]
        else:
            data = np.concatenate((data, dataset[i]), axis=-1)

    std_ = np.std(data, axis=-1).reshape((1, -1, 1))
    mean_ = np.mean(data, axis=-1).reshape((1, -1, 1))
    norm_dataset = (dataset - mean_) / std_

    ## for plots
    labels = ["ACC_X-axis", "ACC_Y-axis", "ACC_Z-axis",
              "GYR_X-axis", "GYR_Y-axis", "GYR_Z-axis",
              "MAG_X-axis", "MAG_Y-axis", "MAG_Z-axis"]
    if args.plot_dimension or args.plot_one_sample:
        root_dir = "results/" + dataset_name + "/dataview/"
        create_directory(root_dir)

        samples = []
        y = []
        for key in label_summary:
            np.random.seed(key)
            mask = reYt == key
            rand = np.random.randint(0, len(reYt[mask]), size=1)
            y.append(reYt[mask][rand])
            samples.append(norm_dataset[mask][rand][0])

        if args.plot_dimension:
        # for i in range(len(y)):
            for j in range(samples[0].shape[0]):
                plt.figure()
                plt.plot(range(samples[0].shape[-1]), samples[0][j, :], label=f"class {y[0][0]}")
                plt.plot(range(samples[1].shape[-1]), samples[1][j, :], label=f"class {y[1][0]}")
                plt.plot(range(samples[2].shape[-1]), samples[2][j, :], label=f"class {y[2][0]}")
                plt.plot(range(samples[3].shape[-1]), samples[3][j, :], label=f"class {y[3][0]}")
                plt.legend(loc='best')
                plt.title(f"{labels[j]} sensor for {len(y)} classes")
                file_name = root_dir + f"{labels[j]}_sensor.png"
                plt.savefig(file_name)
            plt.show()
        if args.plot_one_sample:
            for i in range(len(y)):
                plt.figure()
                plt.plot(range(samples[i].shape[-1]), samples[i][0, :], label=f"Sensor {labels[0]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][1, :], label=f"Sensor {labels[1]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][2, :], label=f"Sensor {labels[2]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][3, :], label=f"Sensor {labels[3]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][4, :], label=f"Sensor {labels[4]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][5, :], label=f"Sensor {labels[5]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][6, :], label=f"Sensor {labels[6]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][7, :], label=f"Sensor {labels[7]}")
                plt.plot(range(samples[i].shape[-1]), samples[i][8, :], label=f"Sensor {labels[8]}")
                plt.legend(loc='best')
                plt.title(f"a sample for {y[i][0]} class")
                file_name = root_dir + f"a_sample_{y[i][0]}_class.png"
                plt.savefig(file_name)
            plt.show()

    if args.plot_samples:
        root_dir = "results/" + dataset_name + "/dataview/"
        create_directory(root_dir)
        y = []
        samples = []
        for key in label_summary:
            np.random.seed(key)
            mask = reYt == key
            rand = np.random.randint(0, len(reYt[mask]), size=10)
            y.append(reYt[mask][rand])
            samples.append(norm_dataset[mask][rand])

        for i in range(len(y)):
            plt.figure()
            for j in range(len(y[i])):
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][0, :], label=f"Sensor {labels[0]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][1, :], label=f"Sensor {labels[1]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][2, :], label=f"Sensor {labels[2]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][3, :], label=f"Sensor {labels[3]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][4, :], label=f"Sensor {labels[4]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][5, :], label=f"Sensor {labels[5]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][6, :], label=f"Sensor {labels[6]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][7, :], label=f"Sensor {labels[7]}")
                plt.plot(range(samples[i][j].shape[-1]), samples[i][j][8, :], label=f"Sensor {labels[8]}")

            plt.legend(loc='upper right', bbox_to_anchor=(0.5, -1.0))
            plt.title(f"samples for {y[i][0]} classes")
            file_name = root_dir + f"{y[i][0]}_classes.png"
            plt.savefig(file_name)
        plt.show()

elif dataset_name is "stabilo":
    pass
else: ## UCR Datasets
    root_dir = ""
    dataset_name = "GunPointOldVersusYoung"
    multivartiate = False
    classes_dict = {'1': "Young", '2': "Old"}
    dataset = read_dataset_ts(root_dir, dataset_name, znorm=True,
                              multivartiate=multivartiate)
    # create_directory("/results/GunPointAgeSpan/dataview/")
    plot_sample_per_dimension(root_dir, dataset, dataset_name,
                              classes_dict=classes_dict)      #plot a random sample per dimension
    # plot_samples_each_class(root_dir, dataset, dataset_name)        #plot random samples for each class in train and testset
    plot_dataset(root_dir, dataset, dataset_name,
                 classes_dict=classes_dict)   # Plot the whole dataset


# if __name__ == "__main__":

