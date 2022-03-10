import numpy as np
import torch as th
from torch import nn
from typing import Tuple


## -----------
## --- Own ---
## -----------
from utils import read_dataset_ts, load_model, throw_out_wrong_classified
from models.models import FCN, TCN
from trainhelper.dataset import Dataset

from paperplots.almost_trades import compute_trades_loss

BASEDIR = "./results/"

import pandas as pd
import torch as t
def load_trained_model(dataset_name, model_name, experiment_seed):
    """
    Assume models(.ckp) are under visualization_for_timeseries/results/experiment_seed/...

    for example: ../results/GunPointAgeSpan/FCN_withoutFC/
    under this directory, should have directory (checkpoints), "reports.csv"
    """
    if dataset_name not in ["ElectricDevices", "FordA", "FordB", "GunPointAgeSpan", "NATOPS",
                            "MelbournePedestrian"]:
        raise ValueError("dataset not exists: {}".format(dataset_name))

    if model_name not in ["FCN_withoutFC", "TCN_withoutFC"]:
        raise ValueError("model not exists: {}".format(model_name))

    multivariate = False
    if dataset_name is "NATOPS":
        multivariate = True

    dataset = read_dataset_ts("./", dataset_name=dataset_name,
                              multivariate=multivariate)
    _, test_x, _y, test_y, labels_dict = dataset[dataset_name]
    testset = Dataset(test_x, test_y)

    path_2_parameters = BASEDIR + dataset_name + "/" + model_name + "/" + experiment_seed + "/"
    report = pd.read_csv(path_2_parameters + "reports.csv")
    ## model setting and loading from checkpoint
    if int(report["best_epoch"][0]) >= 100:
        ckp_path = path_2_parameters + "checkpoints/checkpoint_{}.ckp".format(report["best_epoch"][0])
    else:
        ckp_path = path_2_parameters + "checkpoints/checkpoint_0{}.ckp".format(report["best_epoch"][0])

    if model_name is "FCN_withoutFC":
        kernel_size = [int(k) for k in
                       report["kernel_size"][0][1:-1].split(',')]
        ch_out = [int(k) for k in report["Filter_numbers"][0][1:-1].split(',')]
        model = FCN(ch_in=int(testset.data.shape[1]),
                    ch_out=ch_out,
                    dropout_rate=report["dropout_rate"][0],
                    num_classes=report["num_classes"][0],
                    kernel_size=kernel_size,
                    use_fc=report["use_fc"][0],
                    use_pooling=report["use_pooling"][0],
                    input_dim=testset.data.shape)
    elif model_name is "TCN_withoutFC":
        kernel_size = [int(k) for k in
                       report["kernel_size"][0][1:-1].split(',')]  ## the size also should be the same as ch_out
        ch_out = [int(k) for k in report["Filter_numbers"][0][1:-1].split(',')]
        model = TCN(ch_in=int(testset.data.shape[1]),
                    ch_out=ch_out,
                    kernel_size=kernel_size,
                    dropout_rate=report["dropout_rate"][0],
                    use_fc=report["use_fc"][0],
                    use_pooling=report["use_pooling"][0],
                    num_classes=report["num_classes"][0],
                    input_dim=testset.data.shape)
    else:
        raise ValueError("no model is loaded")

    model = load_model(model=model, ckp_path=ckp_path)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    ### Add softmax or not
    ## here is added
    #model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    if t.cuda.is_available():
        model = model.cuda()

    ## clean testsets (throw out wrong classified data)
    clean_data, clean_labels = throw_out_wrong_classified(model=model,
                                                          data=testset.data,
                                                          labels=testset.labels,
                                                          device=device)
    clean_data = t.tensor(clean_data).float().to(device)
    clean_labels = t.tensor(clean_labels)

    return model, clean_data, clean_labels




def mock_load_trained_model(dataset_name, model_name) -> Tuple[nn.Module, th.Tensor, th.Tensor]:
    model = nn.Sequential(nn.Flatten(1, 2), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
    # 64 is the time dimension
    x = th.rand((1024, 4, 64))
    y = th.zeros((1024, 4))
    return model, x, y


def calculate_elementwise_robustness_score(model, x):

    mins = th.min(x, dim=2).values.unsqueeze(2).cpu().detach().numpy()
    maxs = th.max(x, dim=2).values.unsqueeze(2).cpu().detach().numpy()
    robustness = compute_trades_loss(model, x, mins, maxs, epsilon=0.0005).reshape(-1).cpu().detach().numpy()
    return robustness

from paperplots import results_reader

import pickle

FCN_seeds = [1, 5, 0, 4, 0, 2]
TCN_seeds = [0, 0, 0, 11, 2, 0]
experiment_seeds = {}
for model_name in results_reader.MODELS:
    experiment_seeds[model_name] = {}
    if model_name is "FCN_withoutFC":
        for dataset, seed in zip(results_reader.DATASETS, FCN_seeds):
            experiment_seeds[model_name][dataset] = seed
    elif model_name is "TCN_withoutFC":
        for dataset, seed in zip(results_reader.DATASETS, TCN_seeds):
            experiment_seeds[model_name][dataset] = seed
    else:
        raise ValueError("Model doesn't exist")
        
for dataset in results_reader.DATASETS:
    for model_name in results_reader.MODELS:
        print("[Run] {} with {}".format(model_name, dataset))
        some_file_found = False
        robustness = None
        #for experiment_index in range(12):
        seed = f"experiment_{experiment_seeds[model_name][dataset]}"
        try:
            model, x, y = load_trained_model(dataset, model_name, seed)
            xs = th.split(x, 128, 0)
            rs = []
            for x in xs:
                robustness = calculate_elementwise_robustness_score(model, x)
                rs += [robustness]
            robustness = np.concatenate(rs).reshape(-1)
            some_file_found = True
            output_filename = BASEDIR + f"/{dataset}/{model_name}/experiment_{experiment_seeds[model_name][dataset]}/model_robustness"
            np.save(output_filename, robustness)

            del model
            del x
            del y
            #break
        except FileNotFoundError:
            print(f"No File found: {dataset}/{model_name}/experiment_{experiment_seeds[model_name][dataset]}")
        #except:
        #    print(f"Failed to {dataset}/{model_name}/experiment_{experiment_seeds[model_name][dataset]}")





