"""
Use for Gun Point, but it can be generalized to UCR Dataset
"""

## ------------------
## --- Third-Pary ---
## ------------------
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Dict, Tuple
import os
import csv
import sktime
from sktime.utils.data_io import load_from_tsfile_to_dataframe

## ---------------
## --- Methods ---
## ---------------

# Load Data
def read_dataset_ts(root_dir, dataset_name, znorm: bool = True, multivariate: bool = False):
    """
    To read the dataset from UCR2018
    FOR CLASSIFICATION Problem
    refer to:  https://github.com/hfawaz/dl-4-tsc/blob/master/utils/utils.py
    refer also to: https://github.com/alan-turing-institute/sktime/blob/master/examples/loading_data.ipynb

    Parameters
    ----------
    root_dir
    dataset_name

    Returns
    -------
    Datasets in Dictionary
    """
    datasets_dict = {}
    cur_root_dir = root_dir.replace("-temp", "")
    if multivariate:
        root_dir_dataset = cur_root_dir + "data/Multivariate_ts/" + dataset_name + "/"
    else:
        root_dir_dataset = cur_root_dir + "data/Univariate_ts/" + dataset_name + "/"

    train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(root_dir_dataset, dataset_name + "_TRAIN.ts"))

    test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(root_dir_dataset, dataset_name + "_TEST.ts"))

    ## rename columns --> dim_0: 0, dim_1: 1 ....
    train_x.columns = range(train_x.shape[1])
    test_x.columns = range(test_x.shape[1])
    throw_train_x = []
    throw_test_x = []
    if dataset_name not in ["MelbournePedestrian"]:
        for i in range(train_x.shape[1]):
            for j in range(train_x.shape[0]):
                if len(train_x[i].values[j]) != len(train_x[0].values[0]):
                    throw_train_x.append(j)
                elif np.isnan(train_x[i].values[j]).any():
                    throw_train_x.append(j)
            for j in range(test_x.shape[0]):
                if len(test_x[i].values[j]) != len(test_x[0].values[0]):
                    throw_test_x.append(j)
                elif np.isnan(test_x[i].values[j]).any():
                    throw_test_x.append(j)
        ## Type [num_sample, dim, len_sample]
        train_x_array = np.zeros((train_x.shape[0] - len(throw_train_x), train_x.shape[1], len(train_x[0].values[0])))
        test_x_array = np.zeros((test_x.shape[0] - len(throw_test_x), test_x.shape[1], len(test_x[0].values[0])))
        train_y_array = np.zeros((train_x.shape[0] - len(throw_train_x)))
        test_y_array = np.zeros((test_x.shape[0] - len(throw_test_x)))
        train_index = 0
        test_index = 0
        for i in range(train_x_array.shape[1]): ## dimension
            for j in range(train_x_array.shape[0]): ## sample number
                if j not in throw_train_x:
                    train_x_array[j - train_index, i, :] = train_x[i].values[j]
                    train_y_array[j - train_index] = train_y[j]
                else:
                    train_index += 1
            for j in range(test_x_array.shape[0]):
                if j not in throw_test_x:
                    test_x_array[j - test_index, i, :] = test_x[i].values[j]
                    test_y_array[j - test_index] = test_y[j]
                else:
                    test_index += 1
    else:
        for i in range(train_x.shape[1]):
            for j in range(train_x.shape[0]):
                if len(train_x[i][j].values) != len(train_x[0][0].values):
                    throw_train_x.append(j)
                elif np.isnan(train_x[i][j].values).any():
                    throw_train_x.append(j)
            for j in range(test_x.shape[0]):
                if len(test_x[i][j].values) != len(test_x[0][0].values):
                    throw_test_x.append(j)
                elif np.isnan(test_x[i][j].values).any():
                    throw_test_x.append(j)
        ## Type [num_sample, dim, len_sample]
        train_x_array = np.zeros((train_x.shape[0] - len(throw_train_x), train_x.shape[1], len(train_x[0][0].values)))
        test_x_array = np.zeros((test_x.shape[0] - len(throw_test_x), test_x.shape[1], len(test_x[0][0].values)))
        train_y_array = np.zeros((train_x.shape[0] - len(throw_train_x)))
        test_y_array = np.zeros((test_x.shape[0] - len(throw_test_x)))
        train_index = 0
        test_index = 0
        for i in range(train_x_array.shape[1]): ## dimension
            for j in range(train_x_array.shape[0]): ## sample number
                if j not in throw_train_x:
                    train_x_array[j - train_index, i, :] = train_x[i][j].values
                    train_y_array[j - train_index] = train_y[j]
                else:
                    train_index += 1
            for j in range(test_x_array.shape[0]):
                if j not in throw_test_x:
                    test_x_array[j - test_index, i, :] = test_x[i][j].values
                    test_y_array[j - test_index] = test_y[j]
                else:
                    test_index += 1
    print(throw_train_x)
    print(throw_test_x)
    ## Zero-Mean Norm for each feature(each dim)
    if znorm:
        for i in range(train_x_array.shape[1]):
            std_ = np.std(train_x_array[:, i, :])
            mean_ = np.mean(train_x_array[:, i, :])
            if std_ == 0:
                std_ = 1.0
            ## use them for train and test set
            train_x_array = (train_x_array - mean_) / std_
            test_x_array = (test_x_array - mean_) / std_

    ## label (change str to int) and store in list
    labels_dict = {}
    train_y = np.array(train_y_array)
    test_y = np.array(test_y_array)
    ## manage classes to INT
    targets = np.unique(train_y)
    for j, target in enumerate(targets):
        labels_dict[j] = target
        mask = [(i == target) for i in train_y]
        train_y[mask] = int(j)
        mask = [(i == target) for i in test_y]
        test_y[mask] = int(j)
    train_y = np.array(train_y, dtype="i4")
    test_y = np.array(test_y, dtype="i4")
    datasets_dict[dataset_name] = (deepcopy(train_x_array), deepcopy(test_x_array),
                                   deepcopy(train_y), deepcopy(test_y), labels_dict)
    return datasets_dict


def read_dataset(root_dir, dataset_name):
    """
    To read the dataset from UCR2018
    refer to:  https://github.com/hfawaz/dl-4-tsc/blob/master/utils/utils.py
    refer also to: https://github.com/alan-turing-institute/sktime/blob/master/examples/loading_data.ipynb
    Parameters
    ----------
    root_dir
    dataset_name

    Returns
    -------
    Datasets in Dictionary
    """
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    root_dir_dataset = cur_root_dir + '/data/' + dataset_name + '/'
    df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.ts', sep='\t', header=None)

    df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.ts', sep='\t', header=None)

    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values
    x_test = x_test.values

    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())

    return datasets_dict

def throw_out_wrong_classified(model: nn.Module,
                               data: np.ndarray,
                               labels: np.ndarray,
                               device=None):
    """
    Throw out the Wrong Classified Samples,
    This method is used in Evaluation for Visualization Methods, not including Bias of the target
    of the wrong Sample
    Which we consider that the Visualization methods highlight the importance of the sample to be used
    in right classification. Therefore, the importance of the wrong classified samples may causes some bias

    Parameters
    ----------
    model (nn.Module) : the Deep Learning model structure with the loaded weights
    data (np.ndarray) : The data, which are evaluated (mostly use the Testset)
    labels (np.ndarray) : the according labels from the data
    device : Torch CPU or GPU

    Returns
    -------
    new data and labels, which are correctly classified
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    data = t.tensor(data).float().to(device)
    labels = t.tensor(labels)
    print("data shape:", data.shape)
    print("labels shape:", labels.shape)
    print(f"[INFO] [before throwing wrong classified] The number of data :{len(labels)}")
    predicted = t.zeros(labels.shape)
    mask = []
    with t.no_grad():
        i = 0
        for d, l in zip(data, labels):
            d = d.reshape((1, *d.shape))
            l = l.reshape((-1, 1))
            ## Forward pass
            prediction = model(d)
            if labels.shape[-1] != 1 and len(l) != 1:
                ls, l_counts = np.unique(l.cpu().detach().numpy(), return_counts=True)
                ls_order = np.flip(np.argsort(l_counts))
                segment_order = t.flip(t.argsort(prediction, dim=1), dims=(1,)).cpu().detach().numpy()
                if ls[ls_order[0]] != 4: ## in tool tracking class 4 is garbage class (mix all other classes)
                    if segment_order[0][0] == ls[ls_order[0]]:
                        mask.append(i)
                else:
                    if segment_order[0][0] == ls[ls_order[1]]:
                        mask.append(i)
                    elif segment_order[0][1] == ls[ls_order[1]] and segment_order[0][0] == 4:
                        mask.append(i)
            else:
                predicted[i] = t.argmax(prediction, dim=1)
                single_labels = True
            i += 1
    if single_labels:
        print(predicted)
        print(labels)
        mask = labels == predicted
    cleandata = data[mask].detach().cpu().numpy()
    cleanlabels = labels[mask].detach().cpu().numpy()
    print(f"[INFO] [after throwing wrong classified] The number of data :{len(cleanlabels)}")
    return cleandata, cleanlabels

## Get pytorch layers in the model
def get_layers(model):
    """
    Get layers in the model
    Returns
    -------
        Layers of the model in a List
    """
    layers = []
    for module in model.modules():
        add = True
        for layer in module.modules():
            if isinstance(layer, t.nn.Sequential):
                add = False
        if add:
            layers.append(module)
    return layers

## Get Pytorch model weights and bias
def get_model_weights(model: t.nn.Module, reverse: bool = True):
    """
    Parameters
    ----------
    model: nn.Module (the model structure)
    reverse: bool, reverse the lists or not

    Returns
    -------
    layer_names: list, the name of layers.weights or .bias
    model_weights: t.Tensor, weights and bias in tensor
    """

    layer_names = []
    model_weights = []
    ## get trainable parameters
    if model.model_name() in ["TCN", "FCN", "MLP"]:
        for child, child_layer in model.named_children():
            if isinstance(child_layer, nn.Sequential) and child in ["tcn", "fcn"]:
                for name, param in child_layer.named_parameters():
                    # print(name)
                    if name.split('.')[1] != 'batchnorm':
                        name = child + '.' + name
                        layer_names.append(name)
                        model_weights.append(param)
            elif isinstance(child_layer, nn.Sequential) and child in ["mlp"]:
                for name, param in child_layer.named_parameters():
                    layer_names.append(name)
                    model_weights.append(param)
    else:
        for name, param in model.named_parameters():
            # print(name)
            layer_names.append(name)
            model_weights.append(param)
    if reverse:
        layer_names.reverse()
        model_weights.reverse()
    return layer_names, model_weights

## Load Pytorch Model
def load_model(model, ckp_path:str = None,
               randomized: bool = False,
               independent: bool = False,
               idx_layer: int = None,
               use_cuda=True) -> t.nn.Module:
    """
    First suitable for FCN and TCN
    independent and idx_layer are imporant for loading model with untrained weights
    (independent or cascading)

    Parameters
    ----------
    model: nn.Module (the model structure)
    ckp_path: the path to the checkpoint for the model (.ckp)
    randomized: bool, whether randomize the trained weights
    independent: bool, use for randomization of the weights in models
                (independent or cascading)

    idx_layer: int, which layer should be randomized (independent) or
                    until which layer should be randomized (cascading)
    use_cuda: bool, whether GPU or CPU

    Returns
    -------
    model: nn.Module
    idx (int) : until which layer are randomized
    """
    ## model load and setting
    device = t.device('cuda' if use_cuda else 'cpu')
    if ckp_path is None:
        raise ValueError("checkpoint for model not given!")

    print("[DL Model] Load model with Checkpoint(trained weights)")
    model_ckp = t.load(ckp_path, "cuda" if use_cuda else 'cpu')
    model.load_state_dict(model_ckp["state_dict"])

    ### For Sanity Check
    if randomized:
        layer_names, layer_weights = get_model_weights(model=model) ## also contains bias (weights and bias)
        print("[DL Model] Initialize the trained weights")
        model, idx = layer_randomize(model,
                                layer_names,
                                layer_weights,
                                independent=independent,
                                idx_layer=idx_layer)
        model.to(device)
        model.eval()
        return model, idx
    model.to(device)
    model.eval()
    return model

## [Randomization] Randomize the weights of the layers
def layer_randomize(model,
                    names,
                    weights,
                    independent:bool = False,
                    idx_layer:int = None):
    """
    Parameters
    ----------
    model: nn.Module (the model structure)
    names: list, the name of layers.weights or .bias
    weights: list, of tensors which will be randomized
    independent: bool, only this layer when True, otherwise cascading (successive)

    idx_layer: int, use for remember the last layer, which has been randomized

    Returns
    -------
    model: nn.Module, the model with randomized weights and bias
    """
    def weight_bias_initialize(weight, bias):
        ## bias
        fan_in, _ = t.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        new_bias = t.nn.init.uniform_(bias, -bound, bound)
        ## weights
        new_weights = t.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return new_bias, new_weights

    def layer_initialize(model, weights, split_name, idx_layer):
        t.manual_seed(idx_layer)
        model_name = split_name[0]
        if model.model_name() != 'MLP':
            model_step = int(split_name[1])
        if (model.model_name() == 'FCN') or ((model.model_name() == 'TCN') and (split_name[1] == '1')):
            layer_name = split_name[2]
            bias = weights[idx_layer]
            weight = weights[idx_layer + 1]
            idx = idx_layer + 1
            print("[Random Initialization] Re-initialization of Model {} block {} : layer {}".format(
                model_name, model_step, layer_name))
            if layer_name == 'conv1':
                new_bias, new_weights = weight_bias_initialize(weight, bias)
                model._modules[model_name][model_step].conv1.weight.data = new_weights
                model._modules[model_name][model_step].conv1.bias.data = new_bias
            if layer_name == 'batchnorm':
                ## bias
                new_bias = t.nn.init.zeros_(bias)
                ## weights
                new_weights = t.nn.init.ones_(weight)
                model._modules[model_name][model_step].batchnorm.weight.data = new_weights
                model._modules[model_name][model_step].batchnorm.bias.data = new_bias
            if layer_name == 'fc':
                new_bias, new_weights = weight_bias_initialize(weight, bias)
                model._modules[model_name][model_step].fc.weight.data = new_weights
                model._modules[model_name][model_step].fc.bias.data = new_bias

        elif (model.model_name() == 'TCN') and (split_name[1] == '0'):
            network = split_name[2]
            block_num = int(split_name[3])
            layer_name = split_name[-2]
            print("[Random Initialization] Re-initialization of Model {} block {} : Number {} layer {}".format(
                model_name, model_step, block_num, layer_name))
            if layer_name in ['conv1', 'conv2']:
                weight_v = weights[idx_layer]
                weight_g = weights[idx_layer + 1]
                bias = weights[idx_layer + 2]
                idx = idx_layer + 2
                new_bias, new_weight_v = weight_bias_initialize(weight_v, bias)
                new_weight_g = t.nn.init.kaiming_uniform_(weight_g, a=math.sqrt(5))
                if layer_name == 'conv1':
                    model._modules[model_name][model_step].network[block_num].conv1.weight_v.data = new_weight_v
                    model._modules[model_name][model_step].network[block_num].conv1.weight_g.data = new_weight_g
                    model._modules[model_name][model_step].network[block_num].conv1.bias.data = new_bias
                elif layer_name == 'conv2':
                    model._modules[model_name][model_step].network[block_num].conv2.weight_v.data = new_weight_v
                    model._modules[model_name][model_step].network[block_num].conv2.weight_g.data = new_weight_g
                    model._modules[model_name][model_step].network[block_num].conv2.bias.data = new_bias
            else:
                bias = weights[idx_layer]
                weight = weights[idx_layer + 1]
                idx = idx_layer + 1
                new_bias, new_weights = weight_bias_initialize(weight, bias)
                model._modules[model_name][model_step].network[block_num].downsample.weight.data = new_weights
                model._modules[model_name][model_step].network[block_num].downsample.bias.data = new_bias

        elif model.model_name() == 'MLP': ## only fc (linear layer) needs to be randomized
            model_step = int(split_name[0])
            layer_name = split_name[2]
            bias = weights[idx_layer]
            weight = weights[idx_layer + 1]
            idx = idx_layer + 1
            print("[Random Initialization] Re-initialization of Model {} block {} : layer {}".format(
                model_name, model_step, layer_name)
            )
            new_bias, new_weights = weight_bias_initialize(weight, bias)
            if model_step == 4:
                layer_step = int(split_name[1])
                model._modules["mlp"][model_step][layer_step].weight.data = new_weights
                model._modules["mlp"][model_step][layer_step].bias.data = new_bias
            else:
                model._modules["mlp"][model_step].sequential[1].weight.data = new_weights
                model._modules["mlp"][model_step].sequential[1].bias.data = new_bias
        return model, idx

    if independent:
        print("[Independent randomization] Re-initialization")
        name = names[idx_layer]
        print(f"[Independent randomization] {name} this layer should be initialized")
        split_name = name.split('.')
        ## Initialization
        model, idx = layer_initialize(model=model, weights=weights, split_name=split_name,
                                 idx_layer=idx_layer)
        idx += 1

    else:
        name = names[idx_layer]
        print("[Cascading randomization] Re-initialization")
        print(f"[Cascading randomization] Until {name} this layer should be initialized")
        idx = 0
        while idx <= idx_layer:
            name = names[idx]
            split_name = name.split('.')
            ## Initialization
            model, idx = layer_initialize(model=model, weights=weights, split_name=split_name,
                                          idx_layer=idx)
            idx += 1

        ## For TCN
        if split_name[-2] in ['downsample', 'conv2']:
            idx += 1
    return model, idx

## load saliency maps from files
def load_saliencies(path: str, experiments: List[str]):
    ## Load the Saliency Maps
    ## Abs Norm and No Abs Norm
    saliencymaps_abs_norm_list = []
    saliencymaps_no_abs_norm_list = []
    for experiment in experiments:
        # abs_norm_name = path + "lrpmaps_abs_norm_" + experiment + ".npy"
        # no_abs_norm_name = path + "lrpmaps_no_abs_norm_" + experiment + ".npy"
        # abs_norm_name = path + "modsaliencymaps_abs_norm_" + experiment + ".npy"
        # no_abs_norm_name = path + "modsaliencymaps_no_abs_norm_" + experiment + ".npy"
        abs_norm_name = path + "saliencymaps_abs_norm_" + experiment + ".npy"
        no_abs_norm_name = path + "saliencymaps_no_abs_norm_" + experiment + ".npy"
        abs_norm_maps = np.load(abs_norm_name, allow_pickle=True)
        no_abs_norm_maps = np.load(no_abs_norm_name, allow_pickle=True)
        saliencymaps_abs_norm_list.append(abs_norm_maps.item())
        saliencymaps_no_abs_norm_list.append(no_abs_norm_maps.item())
    return saliencymaps_abs_norm_list, saliencymaps_no_abs_norm_list

def clean_saliency_list(models, testset, saliency_list, testsets):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    data = t.tensor(testset.data).float().to(device)
    labels = t.tensor(testset.labels)
    print(f"[INFO] [before throwing wrong classified] The number of data :{len(labels)}")
    new_saliency_list = []
    count = 0
    for model, saliency in zip(models, saliency_list):
        checkset = testsets[count]
        predicted = t.zeros(labels.shape)
        with t.no_grad():
            i = 0
            for d, l in zip(data, labels):
                d = d.reshape((1, *d.shape))
                l = l.reshape((-1, 1))
                ## Forward pass
                prediction = model(d)
                predicted[i] = t.argmax(prediction, dim=1)
                i += 1
        mask = labels == predicted
        for key in saliency.keys():
            if len(saliency[key][mask]) != len(checkset.labels):
                raise ValueError("Saliency maps Not equal to testset length")
            saliency[key] = saliency[key][mask]
        new_saliency_list.append(saliency)
        count += 1
        print(f"[INFO] [after throwing wrong classified] The number of data :{len(saliency[key])}")
    return new_saliency_list

# Plot Data (Dataset)
def plot_sample_per_dimension(root_dir: str, datasets_dict: Dict, dataset_name: str, classes_dict: Dict):
    """Plot a sample in each class per dimension (per plot)"""
    cur_root_dir = root_dir.replace("-temp", "")
    cur_root_dir = cur_root_dir + "results/" + dataset_name + "/dataview/"
    create_directory(cur_root_dir)
    train_x, test_x, train_y, test_y, labels_dict = datasets_dict[dataset_name]
    sets = [("trainset", train_x, train_y), ("testset", test_x, test_y)]

    ## for sample in each class
    classes = np.unique(train_y)
    samples = {}
    plot_sets = []
    for set in sets:
        name, x, y = set
        for c in classes:
            mask = [(i == c) for i in y]
            samples_c = x[mask]
            class_c = y[mask]
            rand_idx = np.random.randint(low=0, high=len(class_c))
            samples[str(c)] = samples_c[rand_idx, :, :]  ## Dict key [Class]
        plot_sets.append((name, samples))

    ## for coloring dimensions
    num_classes = len(np.unique(train_y))
    color_mapping = plt.get_cmap().colors
    cmap = color_mapping[0::256//num_classes]

    ## Structure of the sample [N, D, L] num, dim, Length
    for set in plot_sets:
        name_type, sample = set
        for j in range(train_x.shape[1]): ## dimension
            plt.figure()  ## only one figure pro dimension
            for i, c in enumerate(classes):
                plt.plot(range(train_x.shape[-1]), sample[str(c)][j, :], label=f"class_{str(labels_dict[c])}: "
                                                                               f"{classes_dict[labels_dict[c]]}",
                         c=cmap[i])
            plt.legend()
            plt.title(f"{name_type} for Dimension {j}")
            file_name = f"{dataset_name}_{name_type}_sample_dimension_{j}.png"
            file_name = os.path.join(cur_root_dir, file_name)
            plt.savefig(file_name)
            plt.show()

def plot_samples_each_class(root_dir: str, datasets_dict: Dict, dataset_name: str, num_samples: int = 2):
    """Plot a sample with whole dimensions for each class in a plot (* num_sample) """
    cur_root_dir = root_dir.replace("-temp", "")
    cur_root_dir = cur_root_dir + "results/" + dataset_name + "/dataview/"
    create_directory(cur_root_dir)
    train_x, test_x, train_y, test_y, labels_dict = datasets_dict[dataset_name]
    sets = [("trainset", train_x, train_y), ("testset", test_x, test_y)]

    ## for sample in each class
    classes = np.unique(train_y)
    samples = {}
    plot_sets = []
    for set in sets:
        name, x, y = set
        for c in classes:
            mask = [(i == c) for i in y]
            samples_c = x[mask]
            class_c = y[mask]
            old_rand_idx = -1
            for i in range(num_samples):
                rand_idx = np.random.randint(low=0, high=len(class_c))
                while old_rand_idx == rand_idx:
                    rand_idx = np.random.randint(low=0, high=len(class_c))
                samples[str(c) + "_" + str(i)] = samples_c[rand_idx, :, :]    ## Dict key [Class_numSample]
                old_rand_idx = rand_idx
        plot_sets.append((name, samples))

    ## for coloring dimensions
    num_dimensions = train_x.shape[1]
    color_mapping = plt.get_cmap().colors
    cmap = color_mapping[0::256//num_dimensions]

    ## Structure of the sample [N, D, L] num, dim, Length
    for set in plot_sets:
        name_type, sample = set
        for c in classes:
            for i in range(num_samples):
                plt.figure() ## only one figure pro classs
                for j in range(train_x.shape[1]): ## dimension
                    plt.plot(range(train_x.shape[-1]), sample[str(c) + "_" + str(i)][j, :], label="dim_" + str(j),
                             c=cmap[j])
                plt.legend()
                plt.title(f"{name_type} for Class {labels_dict[c]}")
                file_name = f"{dataset_name}_{name_type}_sample_class_{labels_dict[c]}_{i}.png"
                file_name = os.path.join(cur_root_dir, file_name)
                plt.savefig(file_name)
                plt.show()

def plot_dataset(root_dir: str, datasets_dict: Dict, dataset_name: str, classes_dict: Dict):
    cur_root_dir = root_dir.replace("-temp", "")
    cur_root_dir = cur_root_dir + "results/" + dataset_name + "/dataview/"
    create_directory(cur_root_dir)
    train_x, test_x, train_y, test_y, labels_dict = datasets_dict[dataset_name]

    ## for coloring labels
    num_classes = len(np.unique(train_y))
    color_mapping = plt.get_cmap().colors
    cmap = color_mapping[0::256//num_classes]

    ## Structure [N, D, L] num, dim, length
    for i in range(train_x.shape[1]):
        plt.figure()
        old_classes = []
        for j in range(train_x.shape[0]):
            if train_y[j] not in old_classes:
                plt.plot(range(train_x.shape[-1]), train_x[j, i, :],
                         label=f"class_{labels_dict[train_y[j]]}: "
                               f"{classes_dict[labels_dict[train_y[j]]]}",
                         c=cmap[train_y[j] - 1])
                old_classes.append(train_y[j])
            else:
                plt.plot(range(train_x.shape[-1]), train_x[j, i, :],
                         c=cmap[train_y[j] - 1])
        plt.legend()
        plt.title(f"Trainset_dim_{i}")
        file_name = f"{dataset_name}_trainset_dim_{i}.png"
        file_name = os.path.join(cur_root_dir, file_name)
        plt.savefig(file_name)

        plt.figure()
        testold_classes = []
        for j in range(test_x.shape[0]):
            if test_y[j] not in testold_classes:
                plt.plot(range(test_x.shape[-1]), test_x[j, i, :],
                         label=f"class_{labels_dict[test_y[j]]}: "
                               f"{classes_dict[labels_dict[test_y[j]]]}",
                         c=cmap[test_y[j] - 1])
                testold_classes.append(test_y[j])
            else:
                plt.plot(range(test_x.shape[-1]), test_x[j, i, :],
                         c=cmap[test_y[j] - 1])
        plt.legend()
        plt.title(f"Testset_dim_{i}")
        file_name = f"{dataset_name}_testset_dim_{i}.png"
        file_name = os.path.join(cur_root_dir, file_name)
        plt.savefig(file_name)
    # plt.show()

def plot_dataset_fromDataset(dataset, save_path: str, criterions: Dict, method_name: str):
    """Dataset is from torch"""

    xt = dataset.data
    y = dataset.labels

    if "threshold_count_histogram" in criterions.keys():
        threshold_count = criterions["threshold_count_histogram"]
    if "threshold_highlightpoint" in criterions.keys():
        threshold_highlight = criterions["threshold_highlightpoint"]

    ## for coloring labels
    num_classes = len(np.unique(y))
    color_mapping = plt.get_cmap().colors
    cmap = color_mapping[0::256 // num_classes]

    ## Structure [N, D, L] num, dim, length
    for i in range(xt.shape[1]):
        plt.figure()
        for j in range(xt.shape[0]):
            plt.plot(range(xt.shape[-1]), xt[j, i, :], label=criterions["label_summary"][y[j]],
                     c=cmap[y[j]])
        plt.legend()
        plt.title(f"threshold_highlightpoint {threshold_highlight} thres_count_hist_{threshold_count} dim_{i}")
        file_name = save_path + "/" + f"threshold_highlightpoint_{threshold_highlight}_thres_count_hist_{threshold_count}" \
                                      f"_{method_name}_dim_{i}.png"
        plt.savefig(file_name)

## Data Preprocessing

## resize samples
def interpolate_resize(sample:t.Tensor, size:Tuple) -> t.Tensor:
    """Resize: use interpolation
       sample: Dimensions [Batch x Channels x width ], should be also use for multiple samples in Batch Axis
       size: output spatial size: Tuple
    """
    print("[INFO] Interpolation: Resize the sample(s)")
    print(f"[Original Shape] {sample.shape}")
    sample = F.interpolate(sample, size=size)
    print(f"[Resized Shape] {sample.shape}")
    return sample

def summarize_label(train_label=None, test_label=None):
    if train_label is None:
        train_label = []
    unique_labels = np.unique(train_label + test_label).astype(int)
    num_cls = len(unique_labels)

    train_y_merged = np.empty_like(train_label)
    test_y_merged = np.empty_like(test_label)

    label_mapping = {}
    for i in range(num_cls):
        label_mapping[i] = [unique_labels[i]]
    ## map the labels
    for key, labels in label_mapping.items():
        for label in labels:
            mask = train_label == label
            train_y_merged[mask] = key
            mask = test_label == label
            test_y_merged[mask] = key
        print(f"[INFO] Merged labels {labels} to label {key}")
    values, counts = np.unique(train_y_merged, return_counts=True)
    print(f"[INFO] Finished merging trainset labels, now labels contain {list(values)} with counts {list(counts)}")
    values, counts = np.unique(test_y_merged, return_counts=True)
    print(f"[INFO] Finished merging trainset labels, now labels contain {list(values)} with counts {list(counts)}")

    return train_y_merged.astype(int), test_y_merged.astype(int)

def count_num_of_sample_per_class(train_y=None, test_y=None):
    train_values, train_counts = np.unique(train_y, return_counts=True)
    test_values, test_counts = np.unique(test_y, return_counts=True)
    print(f"[INFO] Training Label {list(train_values)} with counts {list(train_counts)}")
    print(f"[INFO] Testing Label {list(test_values)} with counts {list(test_counts)}")
    return train_values, train_counts, test_values, test_counts

## For Report
def generate_results_csv(criterions: Dict, store_path: str = None):
    """For training model, to save as report.csv"""
    if store_path is None:
        store_path = "../results/" + criterions["Dataset"] + "/" + criterions["Classifier"]
    file = store_path + "/reports.csv"
    i = 0
    while os.path.exists(file):
        file = store_path + f"/reports_{i}.csv"
        i += 1
    with open(file, "w") as f:
        w = csv.DictWriter(f, criterions.keys())
        w.writeheader()
        w.writerow(criterions)
        # for data in criterions:
        #     w.writerow(data)
    print(f"[INFO] Results CSV is created under {store_path}")
    f.close()

def save_logs(criterions: Dict, store_path: str = None):
    """Use for training model, to save the logs
        For kernel size or dilation and Label summary, please look at the Func. (generate_result_csv)
    """

    if store_path is None:
        store_path = "../results/" + criterions["Dataset"] + "/" + criterions["Classifier"]
    train_summary = store_path + "/logs.csv"
    train_metrics = store_path + "/metrics.csv"
    best_model = store_path + "/best_model.csv"
    confusion_matrix = store_path + "/confusion_matrix.csv"
    i = 0
    while os.path.exists(train_summary):
        train_summary = store_path + f"/logs_{i}.csv"
        train_metrics = store_path + f"/metrics_{i}.csv"
        best_model = store_path + f"/best_model_{i}.csv"
        confusion_matrix = store_path + f"/confusion_matrix_{i}.csv"
        i += 1

    ## for train summary
    if criterions["Classifier"] in ["FCN_withoutFC", "FCN", "TCN", "TCN_withoutFC", "TCN_dense",
                                    "FCN_laststep", "TCN_laststep"]:
        df_summary = pd.DataFrame(data=np.zeros((1, 10), dtype=np.float), index=[0],
                                  columns=["Dataset", "Classifier", "num_classes", "batch_size",
                                           "dropout_rate", "learning_rate", "multiply_factor_lr",
                                           "use_fc", "early_stop",
                                           "epochs"])
        # if criterions["Classifier"] in ["TCN", "TCN_withoutFC", "TCN_dense"]:
        #     df_summary["dilation"] = criterions["dilation"]
        # df_summary["Filter_numbers"] = criterions["Filter_numbers"]
        # df_summary["kernel_size"] = criterions["kernel_size"]

    elif criterions["Classifier"] in ["LSTM", "LSTM_dense"]:
        df_summary = pd.DataFrame(data=np.zeros((1, 13), dtype=np.float), index=[0],
                                  columns=["Dataset", "Classifier", "num_classes", "batch_size",
                                           "dropout_rate", "learning_rate", "multiply_factor_lr",
                                           "use_fc", "early_stop", "Hidden_size",
                                           "num_layers", "bidirectional",
                                           "epochs"])
        df_summary["Hidden_size"] = criterions["Hidden_size"]
        df_summary["num_layers"] = criterions["num_layers"]
        df_summary["bidirectional"] = criterions["bidirectional"]

    elif criterions["Classifier"] in ["LSTMInputCell"]:
        df_summary = pd.DataFrame(data=np.zeros((1, 13), dtype=np.float), index=[0],
                                  columns=["Dataset", "Classifier", "num_classes", "batch_size",
                                           "dropout_rate", "learning_rate", "multiply_factor_lr",
                                           "use_fc", "early_stop", "Hidden_size",
                                           "r", "d_a",
                                           "epochs"])
        df_summary["Hidden_size"] = criterions["Hidden_size"]
        df_summary["r"] = criterions["r"]
        df_summary["d_a"] = criterions["d_a"]

    elif criterions["Classifier"] in ["Utime"]:
        df_summary = pd.DataFrame(data=np.zeros((1, 10), dtype=np.float), index=[0],
                                  columns=["Dataset", "Classifier", "num_classes", "batch_size",
                                           "learning_rate", "multiply_factor_lr",
                                           "early_stop", "dilation",
                                           "kernel_size",
                                           "epochs"])
        df_summary["dilation"] = criterions["dilation"]
        # df_summary["Filter_numbers"] = criterions["Filter_numbers"]
        # df_summary["Maxpool_kernels"] = criterions["Maxpool_kernels"]
        df_summary["kernel_size"] = criterions["kernel_size"]
    elif criterions["Classifier"] in ["MLP"]:
        df_summary = pd.DataFrame(data=np.zeros((1, 9), dtype=np.float), index=[0],
                                  columns=["Dataset", "Classifier", "num_classes", "batch_size",
                                           "learning_rate", "multiply_factor_lr",
                                           "use_fc",
                                           "early_stop",
                                           "epochs"])
        df_summary["use_fc"] = criterions["use_fc"]
        # df_summary["Hidden_size"] = criterions["Hidden_size"]
        # df_summary["dropout_rate"] = criterions["dropout_rate"]

    df_summary["Dataset"] = criterions["Dataset"]
    df_summary["Classifier"] = criterions["Classifier"]
    df_summary["num_classes"] = criterions["num_classes"]
    df_summary["batch_size"] = criterions["batch_size"]
    if criterions["Classifier"] not in ["Utime", "MLP"]:
        df_summary["dropout_rate"] = criterions["dropout_rate"]
        df_summary["use_fc"] = criterions["use_fc"]
    df_summary["learning_rate"] = criterions["learning_rate"]
    df_summary["multiply_factor_lr"] = criterions["multiply_factor_lr"]
    df_summary["early_stop"] = criterions["early_stop"]
    df_summary["epochs"] = criterions["epochs"]
    df_summary.to_csv(train_summary, index=False)

    if criterions["Classifier"] not in ["Utime", "TCN_dense", "LSTM_dense"]:
        ## for train metrics
        df_metrics = pd.DataFrame(data=np.zeros((criterions["epochs"], 5), dtype=np.float),
                                  columns=["train_loss", "val_loss", "train_accuracy", "val_accuracy", "f1_score"])
        df_metrics["train_loss"] = criterions["train_loss"]
        df_metrics["val_loss"] = criterions["val_loss"]
        df_metrics["train_accuracy"] = criterions["train_accuracy"]
        df_metrics["val_accuracy"] = criterions["val_accuracy"]
        df_metrics["f1_score"] = criterions["f1_score"]
        df_metrics.to_csv(train_metrics, index=False)

        ## for best model
        df_metrics = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float),
                                  columns=["train_loss", "test_loss", "train_accuracy", "test_accuracy", "f1_score",
                                           "precision", "recall", "best_epoch"])
        df_metrics["best_epoch"] = criterions["best_epoch"]
        df_metrics["train_loss"] = criterions["train_loss"][criterions["best_epoch"] - 1]
        df_metrics["test_loss"] = criterions["testset_loss"]
        df_metrics["train_accuracy"] = criterions["train_accuracy"][criterions["best_epoch"] - 1]
        df_metrics["test_accuracy"] = criterions["testset_acc"]
        df_metrics["precision"] = criterions["precision_bestepoch"]
        df_metrics["recall"] = criterions["recall_bestepoch"]
        df_metrics["f1_score"] = criterions["f1_bestepoch"]
        df_metrics.to_csv(best_model, index=False)
    else:
        ## for train metrics
        df_metrics = pd.DataFrame(data=np.zeros((criterions["epochs"], 4), dtype=np.float),
                                  columns=["train_loss", "val_loss", "train_f1_score", "val_f1_score"])
        df_metrics["train_loss"] = criterions["train_loss"]
        df_metrics["val_loss"] = criterions["val_loss"]
        df_metrics["train_f1_score"] = criterions["train_f1_score_metric"]
        df_metrics["val_f1_score"] = criterions["val_f1_score_metric"]
        # df_metrics["f1_score"] = criterions["f1_score"]
        df_metrics.to_csv(train_metrics, index=False)

        ## for best model
        df_metrics = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float),
                                  columns=["train_loss", "test_loss", "train_f1_score", "test_f1_score", "f1_score",
                                           "precision", "recall", "best_epoch"])
        df_metrics["best_epoch"] = criterions["best_epoch"]
        df_metrics["train_loss"] = criterions["train_loss"][criterions["best_epoch"] - 1]
        df_metrics["test_loss"] = criterions["testset_loss"]
        df_metrics["train_f1_score"] = criterions["train_f1_score_metric"][criterions["best_epoch"] - 1]
        df_metrics["test_f1_score"] = criterions["testset_f1"]
        df_metrics["precision"] = criterions["precision_bestepoch"]
        df_metrics["recall"] = criterions["recall_bestepoch"]
        df_metrics["f1_score"] = criterions["f1_bestepoch"]
        df_metrics.to_csv(best_model, index=False)

    ## for confusion matrix
    df_metrics = pd.DataFrame(data=criterions["confusion_matrix_testset"], index=[np.arange(0, criterions["num_classes"])],
                              columns=[np.arange(0, criterions["num_classes"])])
    df_metrics.to_csv(confusion_matrix, index=False)
    print(f"[INFO] All Results CSV is created under {store_path}")


def create_directory(directory_path):
    """refer to: https://github.com/hfawaz/dl-4-tsc/blob/master/utils/utils.py"""
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path

def create_path(root_dir, dataset_name, classifier_name):
    output_directory = root_dir + "results/" + dataset_name + "/" + classifier_name
    return create_directory(output_directory)

