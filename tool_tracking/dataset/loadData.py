"""
Load Data into seglearn Format
seglearn : https://dmbee.github.io/seglearn/user_guide.html
"""
## -------------------
## --- Third-party ---
## -------------------
import sys
sys.path.append('..')
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import signal
from tabulate import tabulate
from typing import List, Tuple, Dict
from seglearn.base import TS_Data
from seglearn.pipe import Pype
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt

## ---------------------
## --- tool-tracking ---
## ---------------------
from datatools import Tool, Config, DataBunch
from datatools import MeasurementDataReader, MeasurementSeries, Measurement, DataTypes, to_ts_data
from fhgutils import contextual_recarray_dtype, Segment, filter_ts_data
from fhgutils import filter_labels, one_label_per_window, summarize_labels
from fhgutils.utils import most_frequent_label_per_window
from tool_tracking.tool_utils import combine_sensors

## -----------
## --- Own ---
## -----------
from tool_tracking.tool_utils import Dataset
import trainhelper.dataset as ds

# use for relabeling
relabel_dict = {  # Labels are shared across all tools, 'holes' in both pneumatics
    2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5,  # undefined AKA nothing is happening
   14: 6, 25: 7, 28: 8, 29: 9, 30: 10, 38: 11}

"""
Methods:
    load_data,
    sensor_resampling, 
    pipe_segment,
    labels_info,
    filteroutlabels,
    merge_label_in_window,
    summarize_labels,
    split_data,
    plot_sample_and_measurement,
    merge_labels
"""

def load_data(source: str, tool: str, datatypes: List) -> Dict:
    """
    Parameters
    ----------
    source (str) : path for the data
    tool (str) : the tool to be measured
    datatypes (List) : sensor data [ACC, GYR, MAG, MIC]

    Returns
    -------
    Data Dict : A dictionary, which includes the Data and Labels from the measurement
    """
    mdr = MeasurementDataReader(source=source)
    label_mapping = mdr.label_mapping
    data_dict = mdr.query(query_type=Measurement).filter_by(
        Tool == tool,  ## RIVETER
        DataTypes == datatypes
    ).get()
    data_names = list(data_dict.keys())
    print("[INFO] name of the tool measurements:", data_names)
    print("For example")
    print("[INFO] Sensor DataTypes:", data_dict[data_names[0]].data_keys())

    return data_dict

def read_data_npy(data_path, sparse_data: bool = True, znorm: bool = True):
    """
    To read the tool tracking data from npy files

    Parameters
    ----------
    data_path (str): the path to the directory of data files
    sparse_data (bool): Use sparse labeling data or densely labeling
                        default: True, use sparse labeling
    znorm (bool): use zero normalization or not
                    default: True

    Returns
    -------
    Dataset in Dictionary
    """
    dataset_dict = {}

    if sparse_data:
        train_data = np.load(data_path + '/sparse_label_windowed_traindata.npy')
        train_labels = np.load(data_path + '/sparse_label_windowed_trainlabels.npy')
        test_data = np.load(data_path + '/sparse_label_windowed_testdata.npy')
        test_labels = np.load(data_path + '/sparse_label_windowed_testlabels.npy')
        val_data = np.load(data_path + '/sparse_label_windowed_valdata.npy')
        val_labels = np.load(data_path + '/sparse_label_windowed_vallabels.npy')
    else:
        train_data = np.load(data_path + '/dense_label_windowed_traindata.npy')
        train_labels = np.load(data_path + '/dense_label_windowed_trainlabels.npy')
        test_data = np.load(data_path + '/dense_label_windowed_testdata.npy')
        test_labels = np.load(data_path + '/dense_label_windowed_testlabels.npy')
        val_data = np.load(data_path + '/dense_label_windowed_valdata.npy')
        val_labels = np.load(data_path + '/dense_label_windowed_vallabels.npy')
    if znorm:
        train_data, val_data, test_data = zero_normalization(traindata=train_data, valdata=val_data, testdata=test_data)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def sensors_resampling(Xt, Y, reference: str = "mag"):
    """
    Resample the sensor data to fit them into the same length
    (can be downsampling or upsamp                                                                                                                                                                                                                                  ling)

    Parameters
    ----------
    Xt : The sensor data(array)
    Xc (np.recarray): the Information of the sensor data(Xt)
    Y  : The Labels, which correspond to Xt

    reference (str) : the sensor should be the reference, which other sensors have
                      at the end the same length as the reference sensor

    Returns
    -------
    new Xt , Xc (np.recarray), Y
    """
    num_ts = len(Xt)
    reXt = deepcopy(Xt)
    reY = deepcopy(Y)
    # reXc = np.recarray(shape=(num_ts, ), dtype=contextual_recarray_dtype)

    last_idx_reference = 0
    for i in range(num_ts):
        desc_ = Xt[i].context_data[0][3]
        print(f"[INFO] the original shape of sample is : {Xt[i].ts_data.shape}")
        if desc_ is reference:
            num_update = (i - last_idx_reference)
            sampling_len = Xt[i].ts_data.shape[0]
            for j in range(num_update):
                xt = Xt.ts_data[last_idx_reference + j]
                resample_data = signal.resample(xt, num=sampling_len, axis=0)
                reXt.ts_data[last_idx_reference + j] = resample_data
                reY[last_idx_reference + j] = Y[i]
                # reXt[last_idx_reference + j][:, 0] = Xt[i][:, 0]
                ## Sampling rate should also be changed, because the data are resampled
                reXt[last_idx_reference + j].context_data = Xt[last_idx_reference + j].context_data
                reXt[last_idx_reference + j].context_data[0].sr = Xt[i].context_data[0].sr
                print(f"[INFO] the resampling shape of sample: {reXt.ts_data[last_idx_reference + j].shape}")

            # reXt[i].ts_data = Xt[i].ts_data
            # reXt[i].context_data = Xt[i].context_data
            # reY[i] = Y[i]
            last_idx_reference = i + 1
    return reXt, reY

def sensors_resampling_for_list(Xt: List, Y: List):
    """
        Resample the sensor data to fit them into the same length
        (can be downsampling or upsampling)
        We upsample to mag sampling rate
        Assume: in each list order, data sensor order : [acc, gyr, mag]

        Parameters
        ----------
        Xt (List) : The sensor data(List)
        Y  (List) : The Labels, which correspond to Xt
        Returns
        -------
        new Xt, Y
    """
    num_samples = len(Xt)
    reXt = deepcopy(Xt)
    for i in range(num_samples):
        low_len = Xt[i][0].shape[0]
        refer_len = Xt[i][-1].shape[0]
        if low_len < refer_len:
            resample_data_acc = signal.resample(Xt[i][0], num=refer_len, axis=0)
            resample_data_gyr = signal.resample(Xt[i][1], num=refer_len, axis=0)
            reXt[i][0] = resample_data_acc
            reXt[i][1] = resample_data_gyr
        elif low_len > refer_len:
            print(i)
            print("acc sample length larger than mag sample length")

    return reXt

def none_windowing(Xt, Xc, Yt):
    yt_whole = []
    for yt in Yt:
        diff = np.diff(yt)
        diff = reversed(diff)
        diff = np.append(diff, 0)
        diff = reversed(diff)
        mask = np.where(diff != 0)
        for m in mask:
            pass

def pipe_segment(data: Dict,
                 window_length : float,
                 overlap: float,
                 enforce_size: bool,
                 sensors: List = ['acc', 'gyr', 'mag', 'mic'],
                 discard_time: bool = True,
                 one_array: bool = True):
    """
    fit into seglearn pipe (Xt,Xc,y), resample to make sensor data the same length
    and create windowed data

    Parameters
    ----------
    data (Dict) : Data Dictionary from the Measurement
    window_length (float) : window size
    overlap (float) : the percentage of the overlap during windowing
    enforce_size (bool) : When True, enforce to the same size
    sensors (List) : The sensors ['acc', 'gyr', 'mag', 'mic']
    discard_time (bool) : When True, Time Axis will be thrown away
    one_array (bool) : When True, Combine all the sensors into one array

    Returns
    -------
    If one_array is False :(Xt,Xc,y) with separated parameters (Imu, Mag, Aud)...
    If one_array is True : (Xt,Xc,y) with parameters (Imu, Mag, Aud) in a array
    """

    Xt, Xc, y = to_ts_data(data=data, contextual_recarray_dtype=contextual_recarray_dtype)

    X = TS_Data(Xt, Xc)

    if window_length is not None:
        pipe = Pype([
            ('segment', Segment(window_length=window_length, overlap=overlap, enforce_size=enforce_size,
                                n=len(np.unique(Xc.desc))))
        ])
        X_trans, y_trans = pipe.fit_transform(X, y)
    else:
        sample_x = []
        label_y = []
        num = 0
        for i in range(0, len(y), 3):
            y_values, y_counts = np.unique(y[i], return_counts=True)
            y_diff_acc = np.diff(y[i])
            y_diff_gyr = np.diff(y[i+1])
            y_diff_mag = np.diff(y[i+2])
            start_idx = 0
            for j in range(len(y_diff_acc)):
                if y_diff_acc[j] != 0:
                    end_idx = j + 1
                    sample_acc = X.ts_data[i][start_idx:end_idx, :]
                    sample_gyr = X.ts_data[i+1][start_idx:end_idx, :]
                    sample_x.append([sample_acc, sample_gyr])
                    start_idx = end_idx
                elif j == len(y_diff_acc) - 1:
                    sample_acc = X.ts_data[i][start_idx:, :]
                    sample_gyr = X.ts_data[i+1][start_idx:, :]
                    sample_x.append([sample_acc, sample_gyr])
            start_idx = 0
            for j in range(len(y_diff_mag)): ## mag has different sampling rate
                if y_diff_mag[j] != 0:
                    end_idx = j + 1
                    sample_mag = X.ts_data[i+2][start_idx:end_idx, :]
                    sample_x[num].append(sample_mag)
                    label_y.append(y[i+2][start_idx:end_idx])
                    start_idx = end_idx
                    num += 1
                elif j == len(y_diff_mag) - 1:
                    sample_mag = X.ts_data[i+2][start_idx:, :]
                    if num < len(sample_x):
                        sample_x[num].append(sample_mag)
                        label_y.append(y[i+2][start_idx:])
                        num += 1
    if one_array and window_length is not None:
        ## resampling
        X_trans, y_trans = sensors_resampling(X_trans, y_trans, reference="mag")
    elif one_array and window_length is None:
        X_trans = sensors_resampling_for_list(sample_x, label_y)
        y_trans = label_y

    if one_array:
        ## Must create a Array for all sensor data
        ## Count the total Feature
        dataset = None
        num_features = 0
        for sensor in sensors:
            if sensor in ["acc", "gyr", "mag"]:
                num_features += 3
            elif sensor is "mic":
                num_features += 1
    else:
        Xt_aud, Xc_aud, y_aud = None, None, None
        Xt_acc, Xc_acc, y_acc = None, None, None
        Xt_gyr, Xc_gyr, y_gyr = None, None, None
        Xt_mag, Xc_mag, y_mag = None, None, None

    if window_length is not None:
        for sensor in sensors:
            print(f"[INFO] extract segmented {sensor} data")
            # filter out sensor data (Separately)
            Xt_f, Xc_f, y_f = filter_ts_data(X_trans, y_trans, filt={'desc': [sensor]})
            print(f"[INFO] shape of {sensor} data is", Xt_f.shape)

            if one_array and (dataset is None):
                ## create a size of dataset [#Batch, #Features, #Length]
                dataset = np.zeros((Xt_f.shape[0], num_features, Xt_f[0].shape[0]))

            # discard time column
            for idx in range(Xt_f.shape[0]):
                if discard_time:
                    if one_array:
                        sample = Xt_f[idx][:, 1:].transpose((1, 0))
                    else:
                        sample = [window[:, 1:] for window in Xt_f] ## List

                if one_array:
                    if sensor == "acc":
                        dataset[idx, :3, :] = sample
                        # y_acc = np.array(y_f)
                    elif sensor == "gyr":
                        dataset[idx, 3:6, :] = sample
                        # y_gyr = np.array(y_f)
                    elif sensor == "mag":
                        dataset[idx, 6:9, :] = sample
                        # y_mag = np.array(y_f)
                    elif sensor == "mic":
                        dataset[idx, -1, :] = sample
                        # y_mic = np.array(y_f)
                else:
                    if sensor == 'mic':
                        Xt_aud, Xc_aud, y_aud = Xt_f, Xc_f, y_f
                    elif sensor == 'acc':
                        Xt_acc, Xc_acc, y_acc= Xt_f, Xc_f, y_f
                    elif sensor == 'gyr':
                        Xt_gyr, Xc_gyr, y_gyr = Xt_f, Xc_f, y_f
                    elif sensor == 'mag':
                        Xt_mag, Xc_mag, y_mag = Xt_f, Xc_f, y_f

            if sensor == "acc":
                y_acc = np.array(y_f)
            elif sensor == "gyr":
                y_gyr = np.array(y_f)
            elif sensor == "mag":
                y_mag = np.array(y_f)
    else:
        dataset = []
        y_final = y_trans
        for idx in range(len(X_trans)):
            print(idx)
            sample = np.zeros((num_features, X_trans[idx][0].shape[0]))
            for i in range(len(sensors)):
                if discard_time:
                    sampleX = X_trans[idx][i][:, 1:].transpose((1, 0))
                if i == 0:
                    sample[:3, :] = sampleX
                elif i == 1:
                    sample[3:6, :] = sampleX
                elif i == 2:
                    sample[6:9, :] = sampleX
                else:
                    sample[9:, :] = sampleX
            dataset.append(sample)

    if one_array and window_length is not None:
        if np.isclose(y_acc.all(), y_mag.all()):
            y_final = y_mag
        else:
            y_final = y_acc
    elif not one_array:
        dataset = Dataset(Xt_acc=Xt_acc, Xc_acc=Xc_acc,
                          Xt_gyr=Xt_gyr, Xc_gyr=Xc_gyr,
                          Xt_mag=Xt_mag, Xc_mag=Xc_mag,
                          Xt_aud=Xt_aud, Xc_aud=Xc_aud,
                          y_acc=y_acc, y_gyr=y_gyr,
                          y_mag=y_mag, y_aud=y_aud)
        y_final = None

    ## y list
    return dataset, y_final


### -------------------
### Preprocess the data
### -------------------
def zero_normalization(traindata: np.ndarray, valdata: np.ndarray, testdata: np.ndarray):
    """
    Zero Normalization per each feature

    Parameters
    ----------
    traindata (np.ndarray) : the data with shape [Batch, Feature, Time Length]
    valdata (np.ndarray) : the data with shape [Batch, Feature, Time Length]
    testdata (np.ndarray) : the data with shape [Batch, Feature, Time Length]

    Returns
    -------
    norm_traindata (np.ndarray)
    norm_valdata (np.ndarray)
    norm_testdata (np.ndarray)
    """
    data = None
    for i in range(traindata.shape[0]):
        if data is None:
            data = traindata[i]
        else:
            data = np.concatenate((data, traindata[i]), axis=-1)

    std_ = np.std(data, axis=-1).reshape((1, -1, 1))
    mean_ = np.mean(data, axis=-1).reshape((1, -1, 1))

    norm_traindata = (traindata - mean_) / std_
    norm_valdata = (valdata - mean_) / std_
    norm_testdata = (testdata - mean_) / std_
    return  norm_traindata, norm_valdata, norm_testdata

def labels_info(labels: List):
    values, counts = np.unique(labels, return_counts=True)
    for val, count in zip(values, counts):
        print(f"[INFO] label {val} with {count} samples")


def filteroutlabels(labels: List = [-1], data: Dataset = None, windowing: bool = False):
    """
    Useful for Separate Input [ACC, GYR, MAG ... ]
    Filter the only labels we want to use

    filter out labels [-1]
    also filter out whole windows, in case there is no majority label
    e.g. window length of 5, labels could be [1,1,1,1,0] -> okay. but if [1,1,0,0,2] -> discard window.

    Parameters
    ----------
    labels (List) : the labels that should be filtered out, default: [-1]
    data (Dataset) :  the Dataset Class, which contains Data and Labels

    Returns
    -------
    Thw window with the correct filter
    Xt, Xc, y
    """
    ## sensors Reihenfolge [Acc, gyr, mag, mic]
    if len(labels) != 0:
        print(f"[INFO] Discard all windows containing a garbage label: {labels}")
        if windowing:
            mask = [~np.isin(window, labels).any() for window in data.yt]
            data = data[mask]
        else:
            for i in range(len(data.Xt_acc)):
                mask = [~np.isin(window, labels).any() for window in data.y_acc[i]]
                data.Xt_acc[i] = data.Xt_acc[i][mask]
                data.y_acc[i] = data.y_acc[i][mask]

                mask = [~np.isin(window, labels).any() for window in data.y_gyr[i]]
                data.Xt_gyr[i] = data.Xt_gyr[i][mask]
                data.y_gyr[i] = data.y_gyr[i][mask]

                mask = [~np.isin(window, labels).any() for window in data.y_mag[i]]
                data.Xt_mag[i] = data.Xt_mag[i][mask]
                data.y_mag[i] = data.y_mag[i][mask]

                if data.has_audio():
                    mask = [~np.isin(window, labels).any() for window in data.y_aud[i]]
                    data.Xt_aud[i] = data.Xt_aud[i][mask]
                    data.y_aud[i] = data.y_aud[i][mask]
        return data
    else:
        print("[INFO] No garbage labels given, proceed with next step")
        return data
def filteroutlabels_for_denselytrain(dataset: np.ndarray, labels: np.ndarray,
                                     garbage_labels: List = [-1]):
    """
        Useful for the One Array Input Data (Densely labeling)
        Filter the only labels we want to use (Across labels), like in one sample there are two labels
        if there is only one label, we will filter it out

        all the garbage label will be put into labels -1

        Parameters
        ----------
        dataset (np.ndarray) : The Sensor Data
        labels (np.ndarray) : The correspond Labels
        garbage_labels (List) : The labels that we don't want


        Returns
        -------
        filtered Dataset and Labels
    """
    data_f = []
    labels_f = []
    majority = []
    for i in range(len(labels)):
        ## pruefen fuer Garbage labels
        values, counts = np.unique(labels[i], return_counts=True)
        if len(values) > 1:
            new_labels = labels[i]
            for v in values:
                if v in garbage_labels:
                    labels[i][new_labels == v] = -1
            values, counts = np.unique(labels[i], return_counts=True)
            if len(values) > 1:
                data_f.append(dataset[i])
                labels_f.append(labels[i])
                pos = np.argmax(counts)
                majority.append(values[pos])
    return np.array(data_f), np.array(labels_f), np.array(majority)

def filteroutlabels_one_array(dataset: np.ndarray, labels: np.ndarray,
                              garbage_labels: List = [-1],
                              densely: bool=False):
    """
    Useful for the One Array Input Data
    Filter the only labels we want to use

    filter out labels [-1]
    also filter out whole windows, in case there is no majority label
    e.g. window length of 5, labels could be [1,1,1,1,0] -> okay. but if [1,1,0,0,2] -> discard window.

    Parameters
    ----------
    dataset (np.ndarray) : The Sensor Data
    labels (np.ndarray) : The correspond Labels
    garbage_labels (List) : The labels that we don't want
    densely (bool) : True, if the labels want to be densely labels,
                    False, if the labels want to be windowed labels

    Returns
    -------
    filtered Dataset and Labels
    """
    ## get labels per window
    if not densely:
        y_labels = np.empty([len(labels)])
    else:
        y_rules = []
        y_labels = np.empty((len(labels), len(labels[0])))

    for i in range(len(labels)):
        if not densely:
            values, counts = np.unique(labels[i], return_counts=True)
            idx = np.argmax(counts)
            if counts[idx] > 0.5 * np.sum(counts):
                ## take label only if it occurs more than 50% of the window
                y_labels[i] = int(values[idx])
            else:
                ## else shedule if to be removed
                y_labels[i] = -1
        else:
            mask = np.isin(labels[i], list(garbage_labels))
            if True in mask:
                y_labels[i] = [-1] * len(labels[i])
                y_rules.append(-1)
            else:
                y_labels[i] = labels[i]
                y_rules.append(labels[i][0])
    if not densely:
        mask = np.isin(y_labels, garbage_labels, invert=True) ## mask are only the values that we do consider
    else:
        mask = np.isin(y_rules, garbage_labels, invert=True)

    y_labels = y_labels[mask]
    ## filter
    y_filtered = labels[mask]
    dataset_filtered = dataset[mask]
    print(f"[INFO] original shape Xt: {dataset.shape}")
    print(f"[INFO] filtered shape Xt: {dataset_filtered.shape}")
    return dataset_filtered, y_labels

## degraded ??? ##TODO
def merge_2_one_matrix(dataset: Dataset, sensors: List = ["acc", "gyr", "mag", "mic"]):
    """merge all input features to one Matrix"""
    measure_data = {}
    measuresets = []
    for i in range(len(dataset.Xt_acc)):
        sensor = "acc"
        data = pd.DataFrame(dataset.Xt_acc[i])
        data_ts = data.loc[dataset.y_acc[i] != 8]  ## Label 8 is undefined
        data_target = dataset.y_acc[i][dataset.y_acc[i] != 8]
        measure_data[sensor] = data_ts
        measure_data[sensor + "_y"] = data_target

        sensor = "gyr"
        data = pd.DataFrame(dataset.Xt_gyr[i])
        data_ts = data.loc[dataset.y_gyr[i] != 8]  ## Label 8 is undefined
        data_target = dataset.y_gyr[i][dataset.y_gyr[i] != 8]
        measure_data[sensor] = data_ts
        measure_data[sensor + "_y"] = data_target

        sensor = "mag"
        data = pd.DataFrame(dataset.Xt_mag[i])
        data_ts = data.loc[dataset.y_mag[i] != 8]  ## Label 8 is undefined
        data_target = dataset.y_mag[i][dataset.y_mag[i] != 8]
        measure_data[sensor] = data_ts
        measure_data[sensor + "_y"] = data_target

        if dataset.has_audio():
            sensor = "mic"
            data = pd.DataFrame(dataset.Xt_aud[i])
            data_ts = data.loc[dataset.y_aud[i] != 8]  ## Label 8 is undefined
            data_target = dataset.y_aud[i][dataset.y_aud[i] != 8]
            measure_data[sensor] = data_ts
            measure_data[sensor + "_y"] = data_target

        measureset, fisttimestamp = combine_sensors(measure_data, "acc", sensors[1:])
        measuresets.append(measureset)

    return measuresets

def relabel_action(dataset: List):
    """use here only for non-windowing dataset
        to relabel like 2->0, 3->1 ....
    """
    relabel = np.vectorize(lambda x: relabel_dict[x])
    for i in range(len(dataset)):
        dataset[i]["label"] = relabel(dataset[i]["label"])
        print("measurement_{} have {}".format(i, np.unique(dataset[i]["label"])))
    return dataset

def merge_label_in_window(y: List = None):
    """
    flatten labels of windows to the majority label
    [1,1,1,1,2] -> 1. bad if you've got [0,0,0,0,0,1,1,1,1,1,1] -> 1, this creates anomalous samples of class 1

    Parameters
    ----------
    y: Labels in List

    Returns
    -------
    new filtered labels
    """
    print("pre:", y[2])
    y_f = one_label_per_window(y=y)
    print("post:", y_f[2])
    return y_f

def merge_label_4_dataframe(y: List):
    """use also only for non-windowing dataset"""
    y_labels = []
    for i in range(len(y)):
        values, counts = np.unique(y[i].values, return_counts=True)
        idx = np.argmax(counts)
        # take label only if it occurs more than 50% of the window
        if counts[idx] > 0.5 * np.sum(counts):
            y_labels.append(int(values[idx]))

    print("flattened %i labels: %s" % (len(np.unique(y_labels)), str(np.unique(y_labels))))
    return y_labels

def count_labels(y: List):
    y_labels = np.unique(y)
    label_idx = {}
    for label in y_labels:
        idx = np.where(y == label)
        # label_idx[label].append(idx[0])
        print(f"number of label {label} : {len(idx[0])}")
    return y_labels

def summarize_labels(data: Dataset, summary_labels: dict = None, window_length: float = None, threshold: float = 0.6):
    """
    Summarize the labels in the target vector y according to the summary_labels dictionary and compute the most
    frequent label per window.
    Discard every window where the percentage of the most frequent label does not exceed the given threshold.
    Discard every window labeled with -1 (i.e. garbage).
    """
    if summary_labels is not None and window_length is not None:
        print("[INFO] Summarize labels according to summary_labels dict")
        y_summarized = [merge_labels(el, label_mapping=summary_labels, print_info=True) for el in data.yt]
        data.yt = np.array(y_summarized)
    else:
        data.yt = np.array(data.yt)

    # if window size is not None compute the most frequent label per window and save it in data.y
    # if window size is None only one label exists for that given window
    if window_length is not None:
        print("[INFO] Compute most frequent label per window")
        print(f"[INFO] Discard all windows where the percentage of the most frequent label does not exceed"
              f"{threshold}")
        # if the most frequent label does not exceed the threshold -1 is assigned to that window
        ## one label per window
        data.y = most_frequent_label_per_window(data.yt, threshold=threshold)
    else:
        data.y = data.yt

    print("[INFO] Discard all windows with label -1")
    mask = np.isin(data.y, -1, invert=True)
    values, counts = np.unique(data.y[mask], return_counts=True)
    print(f"[INFO] Unique window labels are {values} with counts {counts}")

    data_masked = data[mask]
    return data_masked

## degraded ##TODO
def split_2_classes(data: List):
    """use only for non-windowing case
        not enough points, do zero padding
    """
    dataset = []
    data_out = []
    y_out = []
    maxlen = 0
    for i in range(len(data)):
        length = len(data[i])
        base = 0
        for j in range(length):
            if data[i].iloc[j, 0] - data[i].iloc[base, 0] >= 2.0:
                sample = data[i].iloc[base: j, :]
                base = j
                if len(sample) > 200:
                    if len(sample) > maxlen:
                        maxlen = len(sample)
                    dataset.append(sample)
    for i in range(len(dataset)):
        data = np.zeros((maxlen, dataset[i].shape[1] - 2))
        data[:len(dataset[i]), :] = dataset[i].iloc[:, 1:-1]
        data_out.append(data)
        y_out.append(dataset[i].iloc[:, -1])
    return data_out, y_out

def balance_classes_oneinput(dataset: Tuple, sampling: str = "up", target_class: int = 1):
    """
    Either Over-Sampling or Up-sampling or Down sampling
    to balance the classes
    Over-sampling: Use the over-sample minority classes (SMOTE) -> the Synethetic Minority Oversampling Technique
                (https://imbalanced-learn.org/stable/over_sampling.html#smote-adasyn)
    Up-sampling : use sklearn resample

    Parameters
    ----------
    dataset (Tuple) : Contain data and labels (Data (np.ndarray) , Labels(np.ndarray))
    sampling (str) : "up" or "down" or "over" or "down_min"
    target_class (int) : (This will be used only by sampling "down" )
                        The Target Class. The number of samples from other classes should be
                        the same as the target class.

    Returns
    -------
    dataset
    """
    if sampling not in ["down_min", "down", "over", "up"]:
        raise ValueError("Please enter the correct str (down_min, down, over, up)")

    data, labels = dataset.data, dataset.labels
    values, counts = np.unique(labels, return_counts=True)
    print("[INFO] Label summary before balancing")
    table = [val_count_tuple for val_count_tuple in zip(values, counts)]
    print(tabulate(table, headers=["Label", "Count"]))

    if sampling is "down_min":
        lowest_count = np.min(counts)
        print(f"[INFO] Limiting to {lowest_count} window(s) per label/class (i.e minimum)")

        # create mask that randomly samples down to "lowest_count" windows per label
        np.random.seed(42)
        mask = np.hstack(
            [np.random.choice(np.where(labels == value)[0], lowest_count, replace=False) for value in values]
        )
        print("[INFO] Apply random under-sampling")
        data_balanced = data[mask]
        labels_balanced = labels[mask]

        values, counts = np.unique(labels_balanced, return_counts=True)
        print("[INFO] Label summary after balancing")

        table = [val_count_tuple for val_count_tuple in zip(values, counts)]
        print(tabulate(table, headers=["Label", "Count"]))
    elif sampling is "down":
        mask = values == target_class
        target_count = counts[mask][0]
        print(f"[INFO] Limiting to {target_count} window(s) per label/class (i.e if"
              f" the number of sample are greater than {target_count})")
        # create mask that randomly samples down to "target_count" windows
        np.random.seed(42)
        mask = counts > target_count
        greater_classes = values[mask]
        smaller_classes = values[~mask]

        great_mask = np.hstack(
            [np.random.choice(np.where(labels == value)[0], target_count, replace=False) for value in greater_classes]
        )

        print(f"[INFO] Apply Down-Sampling (Down sampling to {target_count})")
        data_balanced = data[great_mask]
        labels_balanced = labels[great_mask]
        smaller_data = None
        smaller_labels = None
        for cls in smaller_classes:
            smaller_mask = labels == cls
            if smaller_data is None:
                smaller_data = data[smaller_mask]
                smaller_labels = labels[smaller_mask]
            else:
                smaller_data = np.concatenate((smaller_data, data[smaller_mask]), axis=0)
                smaller_labels = np.concatenate((smaller_labels, labels[smaller_mask]), axis=0)

        data_balanced = np.concatenate((data_balanced, smaller_data), axis=0)
        labels_balanced = np.concatenate((labels_balanced, smaller_labels), axis=0)
        values, counts = np.unique(labels_balanced, return_counts=True)
        print("[INFO] Label summary after balancing")

        table = [val_count_tuple for val_count_tuple in zip(values, counts)]
        print(tabulate(table, headers=["Label", "Count"]))

    elif sampling is "over":
        print("[INFO] Apply SMOTE over-sampling")
        smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
        ## need to reshape data from 3d to 2d
        ## each feature iteration
        data_balanced = None
        for feature in range(data.shape[1]):
            data_balancing, labels_balanced = smote.fit_resample(data[:, feature, :], labels)
            if data_balanced is None:
                data_balanced = data_balancing.reshape(data_balancing.shape[0], 1, data_balancing.shape[-1])
                old_labels = labels_balanced
            else:
                data_balancing = data_balancing.reshape(data_balancing.shape[0], 1, data_balancing.shape[-1])
                data_balanced = np.concatenate((data_balanced, data_balancing), axis=1)
                if not np.array_equal(labels_balanced, old_labels):
                    raise ValueError("balanced Labels miss up")
                else:
                    old_labels = labels_balanced

        values, counts = np.unique(labels_balanced, return_counts=True)
        print("[INFO] Label summary after balancing (SMOTE)")

        table = [val_count_tuple for val_count_tuple in zip(values, counts)]
        print(tabulate(table, headers=["Label", "Count"]))

    elif sampling is "up":
        print("[INFO] Apply Sklearn Resample as Up-sampling")
        highest_count = np.max(counts)
        print(f"[INFO] Sampling to {highest_count} window(s) per label/class (i.e maximum)")

        ## create oversampled data and labels
        data_balanced = np.zeros((highest_count*len(counts), data.shape[1], data.shape[2]))
        labels_balanced = np.zeros((highest_count*len(counts)))
        np.random.seed(42)
        for i, label in enumerate(values):
            if i != np.argmax(counts):
                x_ = data[labels == label]
                y_ = labels[labels == label]
                x_oversampled, y_oversampled = resample(x_,
                                                        y_,
                                                        n_samples=highest_count,
                                                        random_state=42)
                data_balanced[highest_count*i:highest_count*(i+1)] = x_oversampled
                labels_balanced[highest_count*i:highest_count*(i+1)] = y_oversampled
            else:
                data_balanced[highest_count * i:highest_count * (i + 1)] = data[labels == label]
                labels_balanced[highest_count * i:highest_count * (i + 1)] = labels[labels == label]

        values, counts = np.unique(labels_balanced, return_counts=True)
        print("[INFO] Label summary after balancing (Upsampling)")

        table = [val_count_tuple for val_count_tuple in zip(values, counts)]
        print(tabulate(table, headers=["Label", "Count"]))

    dataset_balanced = ds.Dataset(data_balanced, labels_balanced)
    return dataset_balanced

def balance_classes(data: Dataset, sensors: List =['acc', 'gyr', 'mag', 'mic']):
    """
    Balance labels by applying random under-sampling

    Returns
    -------
    Dataset
    """
    values, counts = np.unique(data.y, return_counts=True)
    print("[INFO] Label summary before balancing")

    table = [val_count_tuple for val_count_tuple in zip(values, counts)]
    print(tabulate(table, headers=["Label", "Count"]))

    ## get lowest coun of any label
    lowest_count = np.min(counts)
    print(f"[INFO] Limiting to {lowest_count} window(s) per label/class (i.e minimum)")

    # create mask that randomly samples down to "lowest_count" windows per label
    np.random.seed(42)
    mask = np.hstack(
        [np.random.choice(np.where(data.y == value)[0], lowest_count, replace=False) for value in values]
    )

    print("[INFO] Apply random under-sampling")
    data_balanced = data[mask]

    values, counts = np.unique(data_balanced.y, return_counts=True)
    print("[INFO] Label summary after balancing")

    table = [val_count_tuple for val_count_tuple in zip(values, counts)]
    print(tabulate(table, headers=["Label", "Count"]))

    if "acc" in sensors:
        print(f"[INFO] Xt_acc balanced: {len(data_balanced.Xt_acc)} total windows")
    if "gyr" in sensors:
        print(f"[INFO] Xt_gyr balanced: {len(data_balanced.Xt_gyr)} total windows")
    if "mag" in sensors:
        print(f"[INFO] Xt_mag balanced: {len(data_balanced.Xt_mag)} total windows")
    if "mic" in sensors:
        print(f"[INFO] Xt_mic balanced: {len(data_balanced.Xt_aud)} total windows")
    return data_balanced

def split_data(data: Dataset, test_size: float=0.25, sensors: List=['acc', 'gyr', 'mag', 'mic']) -> Tuple[Dataset, Dataset]:
    """
    Split data into train and test set
    """
    def _split_helper(X):
        return train_test_split(X, data.y, random_state=42, stratify=data.y, test_size=test_size)
    traindata = Dataset()
    testdata = Dataset()
    print(f"[INFO] Splitting {sensors} data into train and test set")
    for sensor in sensors:
        if sensor not in ['acc', 'gyr', 'mag', 'mic']:
            raise ValueError(f"unknown sensor {sensor}")
        if sensor == "acc":
            traindata.Xt_acc, testdata.Xt_acc, traindata.y, testdata.y = _split_helper(data.Xt_acc)
            traindata.Xc_acc, testdata.Xc_acc, _, _ = _split_helper(data.Xc_acc)
        if sensor == "gyr":
            traindata.Xt_gyr, testdata.Xt_gyr, traindata.y, testdata.y = _split_helper(data.Xt_gyr)
            traindata.Xc_gyr, testdata.Xc_gyr, _, _ = _split_helper(data.Xc_gyr)
        if sensor == "mag":
            traindata.Xt_mag, testdata.Xt_mag, traindata.y, testdata.y = _split_helper(data.Xt_mag)
            traindata.Xc_mag, testdata.Xc_mag, _, _ = _split_helper(data.Xc_mag)
        if sensor == "mic":
            traindata.Xt_aud, testdata.Xt_aud, traindata.y, testdata.y = _split_helper(data.Xt_aud)
            traindata.Xc_aud, testdata.Xc_aud, _, _ = _split_helper(data.Xc_aud)
    return traindata, testdata

def split_data_oneinput(data: np.ndarray, y: np.ndarray, test_size: float = 0.25):
    """
    Use for Matrix including all input features

    Parameters
    ----------
    data (np.ndarray) : the data, usually with shape [Batch, Features, Length]
    y (np.ndarray) : the labels
    test_size : how much percent for testset

    Returns
    -------
    Trainset and Testset
    """
    print("[INFO] Splitting data into train and test set")
    traindata, testdata, train_y, test_y = train_test_split(data, y, random_state=42, stratify=y,
                                                            test_size=test_size)
    train_y_values, counts = np.unique(train_y, return_counts=True)
    print(f"[INFO] Trainset Labels {train_y_values} with sample size {counts}")
    test_y_values, counts = np.unique(test_y, return_counts=True)
    print(f"[INFO] Testset Labels {test_y_values} with sample size {counts}")
    trainset = (traindata, train_y)
    testset = (testdata, test_y)
    return trainset, testset

def plot_sample_and_measurement(data_dict, Xt, y):
    """
    Parameters
    ----------
    data_dict: should be DataBunch for the best, should be specified to acc, imu, mag...
    Xt: the data after filter window
    y: Labels

    Returns
    -------
    Two plots
    """
    # plot a sample
    plt.figure(figsize=(24, 4))
    plt.title(f"Single window")
    t = Xt[50][:, 0]
    x_win = Xt[50][:, 1:]
    plt.plot(t, x_win)
    plt.xlabel('Time [a.u.]')
    plt.ylabel('sensor values')

    # plot a measurement for acc
    plot_me = data_dict.ts
    t = plot_me[:, 0]
    x_mea = plot_me[:, 1:]
    plt.figure(figsize=(24, 4))
    plt.title("Measurement #01")
    plt.plot(t, x_mea)
    plt.xlabel('Time [a.u.]')
    plt.ylabel('sensor values')
    plt.show()


def merge_labels(y, label_mapping, print_info=False):
    """
    Merge labels in the target vector y according to the label_mapping dictionary.

    Parameters
    ----------
    y : numpy array, shape (n_samples,)
        Array of labels, i.e. target vector.
    label_mapping : dict of {int: int or list of int}
        Dictionary containing the information for summarizing the labels.
        All labels in each list are summarized to the new label being the respective key.
    print_info : bool, optional
        If set to True info messages will be printed.

    Returns
    -------
    y_summarized : numpy array, shape (n_samples,)
        Array of summarized labels, i.e. new target vector.

    Examples
    --------
    >>> y = np.arange(10)
    >>> y
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> label_mapping = {0: [0, 1, 2, 3, 4, 5], 1: [8, 9], 2: [6, 7]}
    >>> merge_labels(y, label_mapping, print_info=True)
    [INFO] Merged labels [0, 1, 2, 3, 4, 5] to label 0
    [INFO] Merged labels [8, 9] to label 1
    [INFO] Merged labels [6, 7] to label 2
    [INFO] Finished merging labels, now y contains labels [0 1 2] with counts [6 2 2]
    array([0, 0, 0, 0, 0, 0, 2, 2, 1, 1])
    """
    mapping = label_mapping.copy()
    # convert mapping dict to dict of {int: list of int} if necessary
    for key, value in mapping.items():
        if type(value) == int:
            mapping[key] = [value]

    # check if all unique labels from target vector occur in mapping dictionary
    unique_labels = set(np.unique(y).astype(int))
    unique_values = {label for label_list in mapping.values() for label in label_list}

    if not unique_labels.issubset(unique_values):
        raise ValueError(
            f'Input dictionary {label_mapping} does not contain all unique labels of original target vector '
            f'{unique_labels}')
    if print_info and not unique_values.issubset(unique_labels):
        print(
            f'[INFO] Input dictionary {label_mapping} contains label(s) not appearing in original target vector (unique '
            f'labels: {unique_labels})')

    # create empty array and fill it with the corresponding labels
    y_merged = np.empty_like(y)

    for key, labels in mapping.items():
        for label in labels:
            mask = y == label
            y_merged[mask] = key
        if print_info:
            print(f"[INFO] Merged labels {labels} to label {key}")

    values, counts = np.unique(y_merged, return_counts=True)

    if print_info:
        print(f"[INFO] Finished merging labels, now y contains labels {list(values)} with counts {list(counts)}")

    return y_merged.astype(int)
