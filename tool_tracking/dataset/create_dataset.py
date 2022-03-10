'''
Use it to create the dataset for tool tracking
Trainset, Validation set and Testset
'''

## ------------------
## --- Third-Party ---
## ------------------
import argparse
import os
import sys
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import numpy as np
import pandas as pd

## ---------------------
## --- Tool-Tracking ---
## ---------------------
from datatools import ACC, GYR, MAG, MIC, POS, VEL

## -----------
## --- own ---
## -----------
from loadData import load_data, pipe_segment, filteroutlabels
# from loadData import summarize_labels, balance_classes, split_data, split_data_oneinput
from loadData import merge_labels, merge_label_in_window, filteroutlabels_one_array, filteroutlabels_for_denselytrain
from trainhelper.dataset import Dataset, DataSplit

def load_rawdata(args, garbage_labels, label_summary):
    ### setting
    root_dir = parentDir + '/'
    dataset_name = args.Dataset_name
    source = parentDir + '/..' + args.Source
    tool = args.Tool

    sensors = ['acc', 'gyr', 'mag']  # 'mic']
    sensors_list = [ACC, GYR, MAG]

    ## set window length and overlap
    one_matrix = args.One_matrix
    densely_labels = args.Densely_labels
    window_length = args.Window_length  # unit in s
    overlap = args.Overlap  # unit in percent

    ## train/test size
    test_size = args.Test_size
    val_size = args.Val_size

    ## -----------------
    ## --- Load Data ---
    ## -----------------
    ## whole dataset from tool in data dict separately
    data_dict = load_data(source, tool=tool, datatypes=sensors_list)  ## each measurement is a DataBunch

    ## windowing, into one_array
    dataset, yt = pipe_segment(data_dict, window_length=window_length, overlap=overlap, enforce_size=True,
                               sensors=sensors, discard_time=True, one_array=one_matrix)

    ## ------------------
    ## --- Preprocess ---
    ## ------------------

    ## filter out garbage labels
    if one_matrix and not densely_labels:
        dataset, yt = filteroutlabels_one_array(dataset, labels=yt, garbage_labels=garbage_labels,
                                                densely=densely_labels)
    elif one_matrix and densely_labels:
        dataset, yt, majority = filteroutlabels_for_denselytrain(dataset=dataset, labels=yt, garbage_labels=garbage_labels)
    else:
        dataset_f = filteroutlabels(labels=garbage_labels, data=dataset)
    ## merge all input features to an matrix
    if one_matrix:
        # measuresets = merge_2_one_matrix(dataset_f, sensors)
        reYt = merge_labels(yt, label_mapping=label_summary, print_info=True)
        # reYt = merge_label_in_window(reYt)
        dataset = Dataset(data=dataset, labels=np.array(reYt))
        ## split dataset into trainset, valset, testset
        if not densely_labels:
            datasplit = DataSplit(dataset=dataset, test_train_split=test_size, val_train_split=val_size,
                                  shuffle=True)
        else:
            datasplit = DataSplit(dataset=dataset, test_train_split=test_size, val_train_split=val_size,
                                  shuffle=True,
                                  densely_labels=densely_labels,
                                  majority_labels=majority)
        # traindata, testdata = split_data_oneinput(dataset, reYt, test_size=test_size)
        # train_x, train_y = traindata
        # test_x, test_y = testdata

        ## number of train and test set before balance
        trainvalues, traincounts = np.unique(datasplit.trainset.labels, return_counts=True)
        valvalues, valcounts = np.unique(datasplit.valset.labels, return_counts=True)
        testvalues, testcounts = np.unique(datasplit.testset.labels, return_counts=True)
        number_of_trainset = [count_tuple for count_tuple in zip(trainvalues, traincounts)]
        number_of_valset = [count_tuple for count_tuple in zip(valvalues, valcounts)]
        number_of_testset = [count_tuple for count_tuple in zip(testvalues, testcounts)]
        print(number_of_trainset)
        print(number_of_valset)
        print(number_of_testset)

    return datasplit

def split2traintestset(data):
    trainset = data.trainset
    testset = data.testset
    valset = data.valset
    return trainset, testset, valset

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='tool_tracking')
    parser.add_argument("--Source", type=str, default='/data/tool-tracking/tool-tracking-data')
    parser.add_argument("--Tool", type=str, default='electric_screwdriver')
    parser.add_argument("--One_matrix", type=bool, default=True)
    parser.add_argument("--Densely_labels", type=bool, default=True)
    parser.add_argument("--Window_length", type=float, default=1.2)
    parser.add_argument("--Overlap", type=float, default=0.5)
    parser.add_argument("--Test_size", type=float, default=0.25)
    parser.add_argument("--Val_size", type=float, default=0.1)
    # parser.add_argument("--Save_to", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data")
    garbage_labels = [-1, 6, 7, 8, 14]
    label_summary = {"0": [2], "1": [3], "2": [4], "3": [5], "4": [-1]}
    datasets = load_rawdata(args=args, garbage_labels=garbage_labels,
                         label_summary=label_summary)
    trainset, testset, valset = split2traintestset(datasets)

    save_to = parentDir + '/../data/tool_tracking_data/'
    if args.Densely_labels:
        np.save(save_to + 'dense_label_windowed_traindata.npy', trainset.data)
        np.save(save_to + 'dense_label_windowed_trainlabels.npy', trainset.labels)
        np.save(save_to + 'dense_label_windowed_testdata.npy', testset.data)
        np.save(save_to + 'dense_label_windowed_testlabels.npy', testset.labels)
        np.save(save_to + 'dense_label_windowed_valdata.npy', valset.data)
        np.save(save_to + 'dense_label_windowed_vallabels.npy', valset.labels)
    elif args.Window_length is None and not args.Densely_labels:
        np.save(save_to + 'sparse_label_traindata.npy', trainset.data)
        np.save(save_to + 'sparse_label_trainlabels.npy', trainset.labels)
        np.save(save_to + 'sparse_label_testdata.npy', testset.data)
        np.save(save_to + 'sparse_label_testlabels.npy', testset.labels)
        np.save(save_to + 'sparse_label_valdata.npy', valset.data)
        np.save(save_to + 'sparse_label_vallabels.npy', valset.labels)
    else:
        np.save(save_to + 'sparse_label_windowed_traindata.npy', trainset.data)
        np.save(save_to + 'sparse_label_windowed_trainlabels.npy', trainset.labels)
        np.save(save_to + 'sparse_label_windowed_testdata.npy', testset.data)
        np.save(save_to + 'sparse_label_windowed_testlabels.npy', testset.labels)
        np.save(save_to + 'sparse_label_windowed_valdata.npy', valset.data)
        np.save(save_to + 'sparse_label_windowed_vallabels.npy', valset.labels)
