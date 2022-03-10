"""
Use for loading data from UCR datasets
"""

## ------------------
## --- Third-Pary ---
## ------------------
import sys
sys.path.append('..')
import numpy as np
import pandas as pd


## -----------
## --- Own ---
## -----------
from utils import read_dataset_ts, plot_dataset, plot_samples_each_class


if __name__ == "__main__":
    root_dir = "../"
    dataset_name = "FordA"
    znorm = True
    data = read_dataset_ts(root_dir=root_dir, dataset_name=dataset_name,
                           znorm=znorm)
    traindata = data[dataset_name][0]
    testdata = data[dataset_name][1]
    train_y = data[dataset_name][2]
    test_y = data[dataset_name][3]
    classes_dict = {'-1': '-1', '1': '1'}

    # plot_dataset(root_dir=root_dir,
    #              datasets_dict=data,
    #              dataset_name=dataset_name,
    #              classes_dict=classes_dict)
    plot_samples_each_class(root_dir=root_dir,
                            datasets_dict=data,
                            dataset_name=dataset_name,
                            num_samples=1)
    print(data)