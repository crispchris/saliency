"""
Use for UCR datasets
for example: Ford
"""

## -------------------
## --- Third-Party ---
## -------------------
import os
import sys
sys.path.append('..')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import numpy as np
import torch as t
import torch.nn as nn
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf

## -----------
## --- Own ---
## -----------
from utils import read_dataset_ts, create_directory
from utils import generate_results_csv, save_logs
from models.models import TCN, FCN
from models.lstm import LSTM
from models.lstm_cellattention import LSTMWithInputCellAttention
from models.resnet import ResNet
from models.mlp import MLP
from trainhelper.stopping import EarlyStoppingCallback_torch as EarlyStoppingCallback
from trainhelper.dataset import Dataset, TrainValSplit
from trainhelper.trainer import Trainer
from trainhelper.plot import plot_model_result

@hydra.main(config_path= fileDir + "/conf", config_name="config")
def train_model(cfg: DictConfig):
    ## -----------------
    ## --- Setting -----
    ## -----------------
    ## regularization
    ## loss L1 or L2
    loss_term = cfg.models.model.loss_term
    loss_lambda = cfg.models.model.loss_lambda
    # Load the dataset
    root_dir = parentDir + "/"
    dataset_name = cfg.data.node.dataset_name
    multivariate_dataset = cfg.data.node.multivariate
    
    dataset = read_dataset_ts(root_dir, dataset_name, 
                              multivariate = multivariate_dataset)
    train_x, test_x, train_y, test_y, label_dict = dataset[dataset_name]

    label_summary = np.unique(list(test_y) + list(train_y))
    num_cls = len(label_summary)

    ## transfer train and test set into Torch Dataset
    trainset = Dataset(train_x, train_y)
    testset = Dataset(test_x, test_y)

    ## --- Train ---
    ## -------------------
    ## --- hyperparams ---
    ## -------------------

    ## ["FCN", "TCN", "ResNet", "FCN_withoutFC", "TCN_withoutFC",
    ## "TCN_laststep", "FCN_laststep",
    ## "LSTM", "LSTMInputCell"]
    selected_model = cfg.models.model.selected_model

    use_fc = cfg.models.model.use_fc
    use_pooling = cfg.models.model.use_pooling
    use_adaptive_lr = cfg.models.model.use_adaptive_lr
    multiply_factor = cfg.models.model.multiply_factor

    one_matrix = True
    batch_size = cfg.models.model.batch_size
    dropout = cfg.models.model.dropout
    lr = cfg.models.model.lr  ## learning rate
    patience = cfg.models.model.patience  ## for early stopping
    if patience != 0:
        earlystopper = 1
    else:
        earlystopper = None

    epochs = cfg.models.model.epochs

    if selected_model in ["FCN_withoutFC", "FCN", "FCN_laststep", "ResNet"]:
        kernel_size = OmegaConf.create(cfg.models.model.kernel_size)
        ch_out = OmegaConf.create(cfg.models.model.ch_out)
    elif selected_model in ["TCN_withoutFC", "TCN", "TCN_laststep"]:
        dilation = OmegaConf.create(cfg.models.model.dilation)  ## should be always same size as ch_out
        kernel_size = OmegaConf.create(cfg.models.model.kernel_size)  ## the size also should be the same as ch_out
        ch_out = OmegaConf.create(cfg.models.model.ch_out)
    elif selected_model in ["LSTM", "LSTMInputCell"]:
        hidden_size = cfg.models.model.hidden_size
        num_layers = cfg.models.model.num_layers
        bidirectional = cfg.models.model.bidirectional
        if selected_model in ["LSTMInputCell"]:
            r = cfg.models.model.r
            d_a = cfg.models.model.d_a
    elif selected_model in ["MLP"]:
        ch_out = OmegaConf.create(cfg.models.model.ch_out)
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
                           "num_classes": num_cls, "label_summary": label_dict}

    elif selected_model in ["TCN_withoutFC", "TCN", "TCN_laststep"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc,
                           "use_pooling": use_pooling,
                           "early_stop": [patience if earlystopper is not None else None],
                           "dilation": dilation,
                           "Filter_numbers": ch_out,
                           "kernel_size": kernel_size,
                           "num_classes": num_cls, "label_summary": label_dict}
    elif selected_model in ["LSTM"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc,
                           "early_stop": [patience if earlystopper is not None else None],
                           "Hidden_size": hidden_size,
                           "num_layers": num_layers,
                           "bidirectional": bidirectional,
                           "num_classes": num_cls,
                           "label_summary": label_dict}
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
                           "label_summary": label_dict}
    elif selected_model in ["MLP"]:
        hyperparameters = {"batch_size": batch_size, "dropout_rate": dropout,
                           "learning_rate": lr,
                           "multiply_factor_lr": multiply_factor if use_adaptive_lr else None,
                           "use_fc": use_fc,
                           "use_pooling": use_pooling,
                           "early_stop": [patience if earlystopper is not None else None],
                           "Hidden_size": ch_out,
                           "num_classes": num_cls, "label_summary": label_dict}

    criterions = cfg.models.model.criterions
    ## set up the directory to store the results
    ckp_directory = root_dir + "results/" + dataset_name + "/" + selected_model + "/"
    if loss_term == "l1":
        ckp_directory += "l1_regularization/loss_{}".format(loss_lambda)
        directory_done = create_directory(ckp_directory)
    elif loss_term == "l2":
        ckp_directory += "l2_regularization/loss_{}".format(loss_lambda)
        directory_done = create_directory(ckp_directory)
    elif loss_term == "dropout" and dropout != 0.0:
        ckp_directory += "dropout_regularization/dropout_{}".format(dropout)
        directory_done = create_directory(ckp_directory)
    elif loss_term == "None":
        ckp_directory += "no_regularization/no_regularization"
        directory_done = create_directory(ckp_directory)
    else:
        raise ValueError("either test loss regularization, dropout or none")


    ## ---------------------
    ## --- model setting ---
    ## ---------------------
    random_seeds = cfg.data.node.random_seeds
    for i in random_seeds:
        t.manual_seed(i)
        np.random.seed(i)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = True

        ## create data indices for train, Validation Test set splits:
        val_train_split = 0.2
        trainvalsplit = TrainValSplit(trainset, val_train_split=val_train_split)

        ## number of train and test set before balance
        trainvalues, traincounts = np.unique(trainvalsplit.trainset.labels, return_counts=True)
        valvalues, valcounts = np.unique(trainvalsplit.valset.labels, return_counts=True)
        testvalues, test_classcounts = np.unique(testset.labels, return_counts=True)
        number_of_trainset = [count_tuple for count_tuple in zip(trainvalues, traincounts)]
        number_of_valset = [count_tuple for count_tuple in zip(valvalues, valcounts)]
        number_of_testset = [count_tuple for count_tuple in zip(testvalues, test_classcounts)]
        print(f"the Test set size: {len(testset)}")
        print(f"the number of each class in Test set: {test_classcounts}")
        ## set the hyperparameters
        hyperparameters["num_train"] = number_of_trainset
        hyperparameters["num_validation"] = number_of_valset
        hyperparameters["num_test"] = number_of_testset

        trainloader, val_loader = trainvalsplit.get_split(batch_size=batch_size,
                                                          num_workers=1)
        testloader = t.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )


        if patience != 0:
            earlystopper = EarlyStoppingCallback(patience=patience)
        else:
            earlystopper = None

        if selected_model in ["FCN_withoutFC", "FCN", "FCN_laststep"]:
            model = FCN(ch_in=int(train_x.shape[1]),
                        ch_out=ch_out,
                        dropout_rate=dropout,
                        num_classes=num_cls,
                        kernel_size=kernel_size,
                        use_fc=use_fc,
                        use_pooling=use_pooling,
                        input_dim=(1, *trainloader.dataset.data[0].shape))

        elif selected_model in ["TCN_withoutFC", "TCN", "TCN_laststep"]:
            model = TCN(ch_in=int(train_x.shape[1]),
                        ch_out=ch_out,
                        kernel_size=kernel_size,
                        dropout_rate=dropout,
                        use_fc=use_fc,
                        use_pooling=use_pooling,
                        num_classes=num_cls,
                        input_dim=(1, *trainloader.dataset.data[0].shape))

        elif selected_model in ["LSTM"]:
            model = LSTM(ch_in=int(train_x.shape[1]),
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=bidirectional,
                         num_classes=num_cls)

        elif selected_model in ["LSTMInputCell"]:
            model = LSTMWithInputCellAttention(ch_in=int(train_x.shape[1]),
                                               hidden_size=hidden_size,
                                               num_classes=num_cls,
                                               rnndropout=dropout,
                                               r=r,
                                               d_a=d_a)

        elif selected_model == "ResNet":
            ## ResNet includes FC as the last layer
            ch_out = [128, 256, 256, 256, 128]
            strides = [1, 1, 1, 1]
            model = ResNet(ch_in=int(train_x.shape[1]),
                           ch_out=ch_out,
                           num_classes=num_cls,
                           kernel_size=kernel_size,
                           stride=strides)
        elif selected_model in ["MLP"]:
            model = MLP(ch_in=int(train_x.shape[1] * train_x.shape[-1]),
                        ch_out=ch_out,
                        dropout_rate=dropout,
                        num_classes=num_cls)

        ## plot the model structure
        # summary(model)
        loss = nn.CrossEntropyLoss()
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
                          test_set=val_loader,
                          norm=loss_term,
                          loss_lambda=loss_lambda,
                          early_stopping_cb=earlystopper,
                          one_matrix=one_matrix,
                          criterions=criterions,
                          root_dir=ckp_directory)

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
        # store_path = root_dir + 'results/' + result_criterions["Dataset"] + "/" + \
        #              result_criterions["Classifier"]
        # path_done = create_directory(store_path)

        num = 0
        newstore_path = ckp_directory + f"/experiment_{num}"
        while os.path.exists(newstore_path):
            num += 1
            newstore_path = ckp_directory + f"/experiment_{num}"
        newstore_path = ckp_directory + f"/experiment_{num - 1}"
        plot_path = newstore_path + "/result_plots/"

        generate_results_csv(result_criterions, store_path=newstore_path)
        save_logs(result_criterions, store_path=newstore_path)
        plot_model_result(criterions=result_criterions, plot_path=plot_path)
        # plot_dataset(root_dir, dataset, dataset_name)

if __name__ == "__main__":
    train_model()
