## ----------------
## --- build-in ---
## ----------------
import sys
sys.path.append('..')

import time, sys, os
from typing import Dict, List, Optional
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import pandas as pd

## ------------------
## --- Third-party---
## ------------------
import numpy as np
from torch.utils.data import Dataset
import torch as t


## ---------------
## --- Methods ---
## ---------------
def combine_sensors(measuredata: Dict, reference_sensor: str, others_types: List[str], firstTimeStamp= np.Inf):
    """Refer to https://github.com/a-hanf/tool-tracking/blob/master/tool_tracking/data-preprocessing.py"""
    """combine dataframes from different sensors.
    Resample all entries to the frequency of the reference_data
    Fills N/As as nearest occurence timewise
    """
    axises = ['_x', '_y', '_z']
    sensors = [reference_sensor] + others_types
    for sensor in sensors:
        measuredata[sensor]['label'] = measuredata[sensor + '_y']
        for i, axis in enumerate(axises):
            measuredata[sensor] = measuredata[sensor].rename({measuredata[sensor].columns[i+1]: sensor + axis}, axis="columns")
        firstTimeStamp = min(firstTimeStamp, measuredata[sensor][0].min()) ## Column 0 is time[s]

    res = measuredata[reference_sensor].copy()
    for type in others_types:
        res = pd.merge_ordered(
            res,
            measuredata[type],
            left_on=0,
            right_on=0,
            # direction="nearest",
        )
        res = res.loc[:, ~res.columns.duplicated()]
        res["label"] = np.where(np.isnan(res["label_x"]), res["label_y"], res["label_x"])
        del res["label_x"]
        del res["label_y"]
    res = res.loc[res["label"] != -1]
    res = res.loc[res["label"] != 8]
    res = res.rename({res.columns[0]: "time"}, axis="columns")
    res["time"] = res["time"] - firstTimeStamp
    res = res.fillna(0)
    return res, firstTimeStamp

## --- Class ---
class Traindata(Dataset):
    """
    Torch Dataset format
    for training and testing data
    """
    def __init__(self, dataset: Dataset):
        self.Xt_acc, self.Xc_acc = dataset.acc
        self.Xt_gyr, self.Xc_gyr = dataset.gyr
        self.Xt_mag, self.Xc_mag = dataset.mag
        self.Xt_aud, self.Xc_aud = dataset.aud
        self.yt = dataset.get_yt()
        self.y = dataset.get_y()
        if self.y is not None:
            print(f"[INFO] there are {self.__len__()} samples in this dataset")

    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        Xt_acc = t.tensor(self.Xt_acc[item])
        Xt_gyr = t.tensor(self.Xt_gyr[item])
        Xt_mag = t.tensor(self.Xt_mag[item])
        if self.has_audio():
            Xt_aud = t.tensor(self.Xt_aud[item])
            Xt = [Xt_acc, Xt_gyr, Xt_mag, Xt_aud]
        # Xt_acc = t.tensor(self._index(self.Xt_acc, item))
        # Xt_gyr = t.tensor(self._index(self.Xt_gyr, item))
        # Xt_mag = t.tensor(self._index(self.Xt_mag, item))
        # Xt_aud = t.tensor(self._index(self.Xt_aud, item))
        else:
            Xt = [Xt_acc, Xt_gyr, Xt_mag]
        y = self._index(self.y, item)
        return Xt, y

    @staticmethod
    def _index(attr, idx):
        if attr is None or len(attr) == 0:
            return None
        else:
            return np.atleast_1d(attr)[idx, ...]

    def has_acc(self):
        return self.Xt_acc is not None and self.Xc_acc is not None
    def has_gyr(self):
        return self.Xt_gyr is not None and self.Xc_gyr is not None
    def has_mag(self):
        return self.Xt_mag is not None and self.Xc_mag is not None
    def has_audio(self):
        return self.Xt_aud is not None and self.Xc_aud is not None

class Dataset:
    def __init__(self, Xt_acc=None, Xc_acc=None, Xt_gyr=None,
                 Xc_gyr=None, Xt_mag=None, Xc_mag=None,
                 Xt_aud=None, Xc_aud=None, yt=None,
                 y=None, y_acc=None, y_gyr=None, y_mag=None,
                 y_aud=None):
        self.Xt_acc = Xt_acc
        self.Xc_acc = Xc_acc
        self.Xt_gyr = Xt_gyr
        self.Xc_gyr = Xc_gyr
        self.Xt_mag = Xt_mag
        self.Xc_mag = Xc_mag
        self.Xt_aud = Xt_aud
        self.Xc_aud = Xc_aud
        self.yt = yt
        self.y = y
        self.y_acc = y_acc
        self.y_gyr = y_gyr
        self.y_mag = y_mag
        self.y_aud = y_aud

    def __getitem__(self, item):
        return Dataset(
            Xt_acc=self._index(self.Xt_acc, item),
            Xc_acc=self._index(self.Xc_acc, item),
            Xt_gyr=self._index(self.Xt_gyr, item),
            Xc_gyr=self._index(self.Xc_gyr, item),
            Xt_mag=self._index(self.Xt_mag, item),
            Xc_mag=self._index(self.Xc_mag, item),
            Xt_aud=self._index(self.Xt_aud, item),
            Xc_aud=self._index(self.Xc_aud, item),
            yt= self._index(self.yt, item),
            y = self._index(self.y, item),
            y_acc=self._index(self.y_acc, item),
            y_gyr=self._index(self.y_gyr, item),
            y_mag=self._index(self.y_mag, item),
            y_aud=self._index(self.y_aud, item))

    @staticmethod
    def _index(attr, idx):
        if attr is None or len(attr) == 0:
            return None
        else:
            return np.atleast_1d(attr)[idx, ...]

    @property
    def acc(self):
        return self.Xt_acc, self.Xc_acc

    @property
    def gyr(self):
        return self.Xt_gyr, self.Xc_gyr

    @property
    def mag(self):
        return self.Xt_mag, self.Xc_mag

    @property
    def aud(self):
        return self.Xt_aud, self.Xc_aud

    def get_yt(self):
        return self.yt

    def get_y(self):
        return self.y

    def has_acc(self):
        return self.Xt_acc is not None and self.Xc_acc is not None
    def has_gyr(self):
        return self.Xt_gyr is not None and self.Xc_gyr is not None
    def has_mag(self):
        return self.Xt_mag is not None and self.Xc_mag is not None
    def has_audio(self):
        return self.Xt_aud is not None and self.Xc_aud is not None


LABELS = np.arange(4)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
class Trainer:
    """
    Torch Trainer for model
    """

    def __init__(self,
                 model,
                 selected_model: str,
                 loss_func,
                 optim=None,  ## optimizer
                 train_set=None,
                 test_set=None,
                 norm='L1',
                 cuda=True,
                 early_stopping_cb=None,
                 writer=None,
                 one_matrix=None):
        """
        Args:
            model: Model to be trained
            selected_model: the name of model
            loss_func: Loss Function
            optim: Optimizer
            train_data: Training Dataset
            test_data: Testing Dataset
            norm: Regularization L1 or L2
            cuda: whether to use the GPU
            early_stopping_cb: The stopping criterion
            writer: Tensorboard
            one_matrix: if there is only one input matrix
        """
        self._model = model
        self._selected_model = selected_model
        self._loss_func = loss_func
        self._optim = optim
        self._trainset = train_set
        self._testset = test_set
        self._norm = norm
        self._cuda = cuda
        self._early_stopping = early_stopping_cb
        self._writer = writer
        self._one_matrix = one_matrix
        self._cwd = os.getcwd()
        if cuda:
            self._model = self._model.cuda()
            self._loss_func = loss_func.cuda()

    def save_checkpoint(self, epoch):
        print(f"[INFO] the current directory is {self._cwd}")
        if not os.path.isdir(self._cwd + "/checkpoints"):
            os.mkdir(self._cwd + "/checkpoints")
        if os.path.isdir(self._cwd + "/checkpoints/" + self._selected_model):
            t.save({'state_dict': self._model.state_dict()},
                   self._cwd + "/checkpoints/" + self._selected_model + '/checkpoint_{:03d}.ckp'.format(epoch))
        else:
            os.mkdir(self._cwd + "/checkpoints/" + self._selected_model)
            t.save({'state_dict': self._model.state_dict()},
                   self._cwd + "/checkpoints/" + self._selected_model + '/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        model_ckp = t.load(
            self._cwd + "/checkpoints/" + self._selected_model + '/checkpoint_{:03d}.ckp'.format(
                epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(model_ckp['state_dict'])

    def train(self, x, y):
        self._optim.zero_grad()
        predicted = self._model(x)
        loss = self._loss_func(predicted,
                               t.tensor(t.reshape(y.to(device), (-1, 1)), dtype=t.long))  ## label should be (len, 1)
        loss.backward()
        self._optim.step()
        ## tensorboard for the gradient histogram
        return loss.item(), predicted

    def train_step3(self, x1, x2, x3, y):
        self._optim.zero_grad()
        predicted = self._model(x1, x2, x3)
        loss = self._loss_func(predicted, t.tensor(t.reshape(y.to(device), (-1, 1)), dtype=t.long))  ## label should be (len, 1)
        loss.backward()
        self._optim.step()
        ## tensorboard for the gradient histogram
        return loss.item(), predicted

    def train_step4(self, x1, x2, x3, x4, y):
        self._optim.zero_grad()
        predicted = self._model(x1, x2, x3, x4)
        loss = self._loss_func(predicted,
                               t.tensor(t.reshape(y.to(device), (-1, 1)), dtype=t.long))  ## label should be (len, 1)
        # lambda1 = 0.5
        # if self._norm is 'L1':
        #     all_conv1_params = t.cat([x.view(-1) for x in self._model.conv1.parameters()])
        #     all_conv2_params = t.cat([x.view(-1) for x in self._model.conv2.parameters()])
        #     all_conv3_params = t.cat([x.view(-1) for x in self._model.conv3.parameters()])
        #     l1_regularization = lambda1 * (
        #         t.norm(all_conv1_params, 1))  # + t.norm(all_conv2_params, 1) + t.norm(all_conv3_params, 1))
        #     loss += l1_regularization
        loss.backward()
        self._optim.step()

        ## tensorboard for the gradient histogram
        return loss.item(), predicted

    def train_epoch(self):
        self._model.train()
        train_losses = []
        correct = 0
        sum_labels = 0
        for i, (data, label) in enumerate(self._trainset, 0):  ## data.size = [batch, length, features]
            if self._one_matrix:
                xt = data.float().to(device).transpose(1, 2)
                label = label.to(device)
                ## Forward pass
                loss, output = self.train(xt, label)
            else:
                xt_acc = data[0].float().to(device).transpose(1, 2) ## data shape = [B, Features, window size]
                xt_gyr = data[1].float().to(device).transpose(1, 2)
                xt_mag = data[2].float().to(device).transpose(1, 2)

                label = label.to(device)
                loss, output = self.train_step3(xt_acc, xt_gyr, xt_mag, label)
                if len(data) == 4:
                    xt_aud = data[3].float().to(device).transpose(1, 2)
                    loss, output = self.train_step4(xt_acc, xt_gyr, xt_mag, xt_aud, label)

            ## calculate the accuracy
            output = t.argmax(output, dim=1)  ## compute the highest probability
            correct += (output == label.reshape(-1, 1)).sum().item()
            sum_labels += len(label)
            train_losses.append(loss)
        avg_loss = np.mean(train_losses)
        avg_acc = correct / sum_labels * 100
        print('[Training] Train Avg. Loss: {}'.format(avg_loss))
        print('-------------------------------------------')
        print('[Training] Train avg accuracy rate: {} %'.format(avg_acc))
        print('-------------------------------------------')
        return avg_loss, avg_acc

    def test(self, x, y):
        predicted = self._model(x)
        loss = self._loss_func(predicted,
                               t.tensor(y.to(device).reshape(-1, 1), dtype=t.long))  ## label should be (len, 1)
        return loss, predicted

    def test_step3(self, x1, x2, x3, y):
        predicted = self._model(x1, x2, x3)
        loss = self._loss_func(predicted,
                               t.tensor(y.to(device).reshape(-1, 1), dtype=t.long))  ## label should be (len, 1)
        return loss, predicted

    def test_step4(self, x1, x2, x3, x4, y):
        predicted = self._model(x1, x2, x3, x4)
        loss = self._loss_func(predicted,
                               t.tensor(y.to(device).reshape(-1, 1), dtype=t.long))  ## label should be (len, 1)
        return loss, predicted

    def test_val(self):
        self._model.eval()
        test_loss = []
        ground_true = None
        predicted_label = None
        correct = 0
        sum_labels = 0
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0): ## data.size = [batch, length, features]
                if self._one_matrix:
                    xt = data.float().to(device).transpose(1, 2)
                    label = label.to(device)
                    ## Forward pass
                    loss, output = self.test(xt, label)
                else:
                    xt_acc = data[0].float().to(device).transpose(1, 2) ## data shape = [B, Features, window size]
                    xt_gyr = data[1].float().to(device).transpose(1, 2)
                    xt_mag = data[2].float().to(device).transpose(1, 2)
                    label = label.to(device)
                    loss, output = self.test_step3(xt_acc, xt_gyr, xt_mag, label)

                    if len(data) == 4:
                        xt_aud = data[3].float().to(device).transpose(1, 2)
                        loss, output = self.test_step4(xt_acc, xt_gyr, xt_mag, xt_aud, label)

                ## calculate the accuracy
                output = t.argmax(output, dim=1)  ## compute the highest probability
                correct += (output == label.reshape(-1, 1)).sum().item()
                sum_labels += len(label)

                test_loss.append(loss)

                ## for f1 score ## collect all predict and label
                if predicted_label is None:
                    predicted_label = output
                else:
                    predicted_label = t.cat((predicted_label, output), dim=0)
                if ground_true is None:
                    ground_true = label
                else:
                    ground_true = t.cat((ground_true, label), dim=0)

            ## f1 score calculation
            f1 = f1_score(ground_true.cpu().numpy(), predicted_label.cpu().numpy(), average='macro')
            avg_loss = t.mean(t.stack(test_loss))
            avg_acc = correct / sum_labels * 100
            print('[Evaluation] Test Avg. Loss: {}'.format(avg_loss))
            print('-------------------------------------------')
            print('[Evaluation] Test avg accuracy rate: {} %'.format(avg_acc))
            print('-------------------------------------------')
            print('[Evaluation] f1 score', f1)

        return avg_loss, avg_acc, f1

    def fit(self, epochs=-1):
        assert self._early_stopping is not None or epochs > 0
        ## metrics
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        f1s = []

        self.epoch_count = 1
        best_loss = np.Inf
        best_epoch = None
        while True:
            if self.epoch_count > epochs:
                metrics, df_cm = self.evaluate(best_epoch=best_epoch)
                break
            print('epoch : {}'.format(self.epoch_count))
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, f1 = self.test_val()

            if test_loss < best_loss:
                best_epoch = self.epoch_count
                best_loss = test_loss
                self.save_checkpoint(best_epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            f1s.append(f1)

            if self._early_stopping is not None:
                self._early_stopping.step(test_loss)
                self._early_stopping.should_stop()
                if self._early_stopping.early_stop is True:
                    print('[EarlyStopping] The best score is on epoch {} and the loss {} %'.format(best_epoch, best_loss))
                    metrics, df_cm = self.evaluate(best_epoch=best_epoch)
                    break
            self.epoch_count += 1

        losses = {"train": train_losses, "test": test_losses}
        accuracy = {"train": train_accs, "test": test_accs}

        return losses, accuracy, metrics, f1s, df_cm

    def evaluate(self, best_epoch):
        print('[Evaluation] best_epoch for loss: {}'.format(best_epoch))
        print('---- confusion matrix for the best epoch acc')
        self.restore_checkpoint(best_epoch)
        self._model.eval()
        labels = None
        predictions = None
        ## f1 score, precision, recall
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0):
                if self._one_matrix:
                    xt = data.float().to(device).transpose(1, 2)
                    label = label.to(device)
                    ## Forward pass
                    predicted = self._model(xt)
                else:
                    xt_acc = data[0].float().to(device).transpose(1, 2) ## data shape = [B, Features, window size]
                    xt_gyr = data[1].float().to(device).transpose(1, 2)
                    xt_mag = data[2].float().to(device).transpose(1, 2)
                    label = label.to(device)
                    predicted = self._model(xt_acc, xt_gyr, xt_mag)

                    if len(data) == 4:
                        xt_aud = data[3].float().to(device).transpose(1, 2)
                        predicted = self._model(xt_acc, xt_gyr, xt_mag, xt_aud)

                ## calculate the accuracy
                predicted = t.argmax(predicted, dim=1)  ## compute the highest probability
                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
                if labels is None:
                    labels = label
                else:
                    labels = t.cat((labels, label), dim=0)

        predictions = np.asarray(predictions.cpu().numpy()).reshape(-1, 1)
        labels = np.asarray(labels.cpu().numpy()).reshape(-1, 1)
        cm = confusion_matrix(labels, predictions, labels=LABELS)
        df_cm = pd.DataFrame(cm, index=[i for i in "0123"],
                             columns=[i for i in "0123"])
        f1_s = f1_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        metrics = {"precision": precision, "recall": recall, "f1": f1_s}
        return metrics, df_cm
