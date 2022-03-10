"""
Use it as main to train deep learning models

Load training set and validation set
create torch DataLoader
load Model
set train parameter
run training
run evaluate
look at result
## fix make (load and save attributes from/to a yaml config file)
"""

## ------------------
## --- Third-Party ---
## ------------------
import os
import sys
sys.path.append('..')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import yaml
import numpy as np
import torch as t
import torch.nn as nn
from torchsummary import summary
from sklearn.utils import compute_class_weight
import hydra
from omegaconf import DictConfig, OmegaConf

## -----------
## --- own ---
## -----------
from utils import generate_results_csv, save_logs, create_directory
from dataknowing.loadData import read_data_npy
from dataknowing.loadData import balance_classes_oneinput

from models.tcn_layer import TCN_layer
from models.models import TCN_4base, FCN_4baseline, TCN_3base, FCN_3baseline
from models.unet import Utime
from models.models import TCN, TCN_dense, FCN
from models.resnet import ResNet
from models.lstm import LSTM, LSTM_dense
from models.lstm_cellattention import LSTMWithInputCellAttention
from models.tcn import TemporalConvNet
from trainhelper.trainer import Trainer
from trainhelper.cross_entropy_with_dice_loss import CrossEntropy_GDice_Loss
from trainhelper.stopping import EarlyStoppingCallback
from trainhelper.dataset import Dataset, DataSplit
from trainhelper.plot import plot_model_result


@hydra.main(config_path="conf", config_name="config")
def train_model(cfg: DictConfig):
    ## -----------------
    ## --- Setting -----
    ## -----------------
    root_dir = parentDir + "/"
    dataset_name = cfg.node.dataset_name
    data_path = cfg.node.data_path
    ## set window length and overlap
    one_matrix = cfg.node.one_matrix
    znorm = cfg.node.znorm ## zero normalization
    balance = cfg.node.balance
    densely_labels = cfg.node.densely_labels
    window_length = cfg.node.window_length # unit in s
    # overlap = cfg.node.overlap # unit in percent

    ## -----------------
    ## --- Load Data ---
    ## -----------------
    ## whole dataset from tool in data dict separately
    data_path = parentDir + '/' + data_path

    sparse_labels = False if densely_labels else True
    trainset, valset, testset = read_data_npy(data_path=data_path,
                                              sparse_data=sparse_labels,
                                              znorm=znorm)
    trainset = Dataset(data=trainset[0], labels=trainset[1])
    valset = Dataset(data=valset[0], labels=valset[1])
    testset = Dataset(data=testset[0], labels=testset[1])

    ## number of train and test set before balance
    trainvalues, traincounts = np.unique(trainset.labels, return_counts=True)
    valvalues, valcounts = np.unique(valset.labels, return_counts=True)
    testvalues, testcounts = np.unique(testset.labels, return_counts=True)
    number_of_trainset = [count_tuple for count_tuple in zip(trainvalues, traincounts)]
    number_of_valset = [count_tuple for count_tuple in zip(valvalues, valcounts)]
    number_of_testset = [count_tuple for count_tuple in zip(testvalues, testcounts)]
    print(number_of_trainset)
    print(number_of_valset)
    print(number_of_testset)
    label_summary = trainvalues

    ## TODO Downsampling (not only random)
    if balance:
        trainset = balance_classes_oneinput(trainset, sampling="down", target_class=1)
        # trainset = balance_classes_oneinput(trainset, sampling="over")
        balance_y = trainset.labels
        balance_trainvalues, balance_traincounts = np.unique(balance_y, return_counts=True)
        number_of_balancetrainset = [count_tuple for count_tuple in zip(balance_trainvalues,
                                                                        balance_traincounts)]
        class_weights = None
    else:
        number_of_balancetrainset = None
        label_counts = trainset.labels.reshape(-1)
        class_weights = compute_class_weight('balanced', np.unique(trainset.labels),
                                             label_counts)
    ## set to pytorch dataset
    # trainset = Dataset(traindata[0], traindata[1])
    # testset = Dataset(testdata[0], testdata[1])

    ## -----------------------------------------------------------------------------
    ## --- Train ---
    ## -------------------
    ## --- hyperparams ---
    ## -------------------
    ## ["BaselineFCN", "FCN", "TCN", "ResNet", "FCN_withoutFC", "TCN_withoutFC",
    ##  "LSTM", "LSTMInputCell"]
    ## For densely Labeling
    ## ["TCN", Utime"]
    selected_model = cfg.model.selected_model
    num_cls = len(label_summary)

    if selected_model not in ["Utime", "TCN_dense"]:
        use_fc = cfg.model.use_fc
    if selected_model in ["TCN", "FCN", "TCN_withoutFC", "FCN_withoutFC", "TCN_laststep", "FCN_laststep"]:
        use_pooling = cfg.model.use_pooling

    use_adaptive_lr = cfg.model.use_adaptive_lr
    multiply_factor = cfg.model.multiply_factor
    batch_size = cfg.model.batch_size
    val_batch_size = cfg.model.val_batch_size
    dropout = cfg.model.dropout
    lr = cfg.model.lr  ## learning rate
    patience = cfg.model.patience ## for early stopping
    if patience != 0:
        earlystopper = 1
    else:
        earlystopper = None

    epochs = cfg.model.epochs

    if selected_model in ["FCN_withoutFC", "FCN", "FCN_laststep", "ResNet"]:
        kernel_size = OmegaConf.create(cfg.model.kernel_size)
        ch_out = OmegaConf.create(cfg.model.ch_out)
    elif selected_model in ["TCN_withoutFC", "TCN", "TCN_dense", "TCN_laststep"]:
        dilation = OmegaConf.create(cfg.model.dilation) ## should be always same size as ch_out
        kernel_size = OmegaConf.create(cfg.model.kernel_size) ## the size also should be the same as ch_out
        ch_out = OmegaConf.create(cfg.model.ch_out)
    elif selected_model in ["LSTM", "LSTMInputCell", "LSTM_dense"]:
        hidden_size = cfg.model.hidden_size
        num_layers = cfg.model.num_layers
        bidirectional = cfg.model.bidirectional
        if selected_model in ["LSTMInputCell"]:
            r = cfg.model.r
            d_a = cfg.model.d_a
    elif selected_model in ["Utime"]:
        ch_out = OmegaConf.create(cfg.model.ch_out)
        kernel_size = cfg.model.kernel_size
        maxpool_kernels = OmegaConf.create(cfg.model.maxpool_kernels)
        dilation = cfg.model.dilation
    else:
        kernel_size = None

    # Dict for Hyper parameters
    if selected_model in ["FCN_withoutFC", "FCN", "FCN_laststep", "ResNet"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc,
                           "use_pooling": use_pooling,
                           "early_stop": [patience if earlystopper is not None else None],
                           "Filter_numbers": ch_out,
                           "kernel_size": kernel_size,
                           "num_classes": num_cls, "label_summary": label_summary}

    elif selected_model in ["TCN_withoutFC", "TCN", "TCN_dense", "TCN_laststep"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc if selected_model not in ["TCN_dense"] else None,
                           "use_pooling": use_pooling if selected_model not in ["TCN_dense"] else None,
                           "early_stop": [patience if earlystopper is not None else None],
                           "dilation": dilation,
                           "Filter_numbers": ch_out,
                           "kernel_size": kernel_size,
                           "num_classes": num_cls, "label_summary": label_summary}

    elif selected_model in ["LSTM", "LSTM_dense"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc,
                           "early_stop": [patience if earlystopper is not None else None],
                           "Hidden_size": hidden_size,
                           "num_layers": num_layers,
                           "bidirectional": bidirectional,
                           "num_classes": num_cls,
                           "label_summary": label_summary}

    elif selected_model in ["LSTMInputCell"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc,
                           "early_stop": [patience if earlystopper is not None else None],
                           "Hidden_size": hidden_size,
                           "d_a": d_a,
                           "r": r,
                           "num_classes": num_cls,
                           "label_summary": label_summary}

    elif selected_model in ["Utime"]:
        hyperparameters = {"batch_size": batch_size,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "early_stop": [patience if earlystopper is not None else None],
                           "dilation": dilation,
                           "Filter_numbers": ch_out,
                           "Maxpool_kernels": maxpool_kernels,
                           "kernel_size": kernel_size,
                           "num_classes": num_cls,
                           "label_summary": label_summary}

    hyperparameters["num_train"] = number_of_trainset
    hyperparameters["num_validation"] = number_of_valset
    hyperparameters["num_test"] = number_of_testset
    hyperparameters["num_balanced_train"] = number_of_balancetrainset

    criterions = cfg.model.criterions
    ## ---------------------
    ## --- model setting ---
    ## ---------------------
    random_seeds = cfg.node.random_seeds
    for i in random_seeds:
        t.manual_seed(i)
        np.random.seed(i)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = True

        ## set to pytorch dataset
        trainloader = t.utils.data.DataLoader(trainset,
                                              batch_size= batch_size,
                                              shuffle= True)
        valloader = t.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             shuffle=False)
        testloader = t.utils.data.DataLoader(testset,
                                             batch_size= batch_size,
                                             shuffle= False)
        ## early stop
        if patience != 0:
            earlystopper = EarlyStoppingCallback(patience=patience)
        else:
            earlystopper = None

        if selected_model in ["FCN_withoutFC", "FCN", "FCN_laststep"]:
            model = FCN(ch_in=int(trainset.data.shape[1]),
                        ch_out=ch_out,
                        dropout_rate=dropout,
                        num_classes=num_cls,
                        kernel_size=kernel_size,
                        use_fc=use_fc,
                        use_pooling=use_pooling,
                        input_dim=(1, *trainloader.dataset.data[0].shape))

        elif selected_model in ["TCN_withoutFC", "TCN", "TCN_laststep"]:
            model = TCN(ch_in=int(trainset.data.shape[1]),
                        ch_out=ch_out,
                        kernel_size=kernel_size,
                        dropout_rate=dropout,
                        use_fc=use_fc,
                        use_pooling=use_pooling,
                        num_classes=num_cls,
                        input_dim=(1, *trainloader.dataset.data[0].shape))

        elif selected_model in ["TCN_dense"]:
            model = TCN_dense(ch_in=int(trainset.data.shape[1]),
                              ch_out=ch_out,
                              kernel_size=kernel_size,
                              dropout_rate=dropout,
                              num_classes=num_cls)

        elif selected_model in ["LSTM"]:
            model = LSTM(ch_in=int(trainset.data.shape[1]),
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=bidirectional,
                         num_classes=num_cls)

        elif selected_model in ["LSTM_dense"]:
            model = LSTM_dense(ch_in=int(trainset.data.shape[1]),
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               num_classes=num_cls)

        elif selected_model in ["LSTMInputCell"]:
            model = LSTMWithInputCellAttention(ch_in=int(trainset.data.shape[1]),
                                               hidden_size=hidden_size,
                                               num_classes=num_cls,
                                               rnndropout=dropout,
                                               r=r,
                                               d_a=d_a)

        elif selected_model in ["ResNet"]:
            ## ResNet includes FC as the last layer
            ch_out = [128, 256, 256, 256, 128]
            strides = [1, 1, 1, 1]
            model = ResNet(ch_in=int(trainset.data.shape[1]),
                           ch_out=ch_out,
                           num_classes=num_cls,
                           kernel_size=kernel_size,
                           stride=strides)

        elif selected_model in ["Utime"]:
            model = Utime(ch_in=int(trainset.data.shape[1]),
                          ch_out=ch_out,
                          maxpool_kernels=maxpool_kernels,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          num_classes=num_cls)

        else:
            if selected_model is "TCN_CAM_nosigmoid":
                tcn_mag = TCN_layer(ch_in=3, dilation=[1, 2, 4])
                tcn_gyr = TCN_layer(ch_in=3, dilation=[1, 2, 4])
                tcn_acc = TCN_layer(ch_in=3, dilation=[1, 2, 4])
                if trainset.has_audio():
                    tcn_aud = TCN_layer(ch_in=1, dilation=[1, 2, 4])
                    # concat inputs
                    model = TCN_4base(tcn_acc, tcn_gyr, tcn_mag, tcn_aud, dropout_rate=dropout,
                                    num_classes=num_cls)
                else:
                    model = TCN_3base(tcn_acc, tcn_gyr, tcn_mag, dropout_rate=dropout,
                                    num_classes=num_cls)
            if selected_model is "BaselineFCN":
                # Fully Convolutional inputs
                if trainset.has_audio():
                    model = FCN_4baseline(in_mag=3, in_acc=3, in_gyr=3, in_aud=1, num_classes=num_cls)
                else:
                    model = FCN_3baseline(in_mag=3, in_acc=3, in_gyr=3, num_classes=num_cls)

        ## plot the model structure
        # summary(model)

        ## ----------------
        ## --- Training ---
        ## ----------------

        # loss = nn.NLLLoss(reduction='mean')
        if not densely_labels:
            loss = nn.CrossEntropyLoss()
        else:
            loss = CrossEntropy_GDice_Loss(apply_nonlin=None,
                                           smooth=1e-5,
                                           cross_en_weight=0.6)

        optimizer = t.optim.Adam(params=model.parameters(), lr=lr)
        scheduler = None
        if use_adaptive_lr:
            scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=multiply_factor,
                                                               patience=30)

        trainer = Trainer(dataset_name=dataset_name,
                          model=model,
                          selected_model=selected_model,
                          loss_func=loss,
                          optim=optimizer,
                          scheduler=scheduler,
                          train_set=trainloader,
                          test_set=valloader,
                          early_stopping_cb=earlystopper,
                          one_matrix=one_matrix,
                          criterions=criterions,
                          root_dir=root_dir,
                          class_weights=class_weights)

        ## store all result in criterions, and also Dataframe for Confusion matrix
        result_criterions, df_cm = trainer.fit(epochs=epochs)

        ### Evaluation on the Testset
        testset_acc, testset_cm = trainer.evaluate_model(dataset=testloader,
                                                         best_epoch=result_criterions["best_epoch"])

        for key, value in hyperparameters.items():
            result_criterions[key] = value
        ## add epoch as criterions
        ## assume loss is always as criterion
        result_criterions["epochs"] = len(result_criterions["train_loss"])

        ## save the results as report and logs
        store_path = root_dir + 'results/' + result_criterions["Dataset"] + "/" + \
                        result_criterions["Classifier"]
        path_done = create_directory(store_path)
        num = 0
        newstore_path = store_path + f"/experiment_{num}"
        while os.path.exists(newstore_path):
            num += 1
            newstore_path = store_path + f"/experiment_{num}"
        newstore_path = store_path + f"/experiment_{num-1}"
        plot_path = newstore_path + "/result_plots/"

        generate_results_csv(result_criterions, store_path=newstore_path)
        save_logs(result_criterions, store_path=newstore_path)
        plot_model_result(criterions=result_criterions, plot_path=plot_path)


        # else:
        #     trainer = Trainer(model, selected_model, loss, optimizer, trainloader, testloader,
        #                       early_stopping_cb= earlystopper, one_matrix=one_matrix)
        #     losses, accuracy, metrics, fls, df_cm = trainer.fit(epochs= epochs)
        #
        #     train_result = [{
        #         "loss": losses,
        #         "accuracy": accuracy,
        #         "metrics": metrics,
        #         "f1_score" : fls,
        #         "confusion_matrix": df_cm
        #     }]
        #
        #     with open("train_result_cam.yaml", "w") as yamlfile:
        #         data = yaml.dump(train_result, yamlfile)
        #         print("Write successful")


if __name__ == "__main__":
    train_model()
