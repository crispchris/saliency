"""
Pytorch Deep Learning Model trainer
For densely labeling
"""
## ----------------
## --- build-in ---
## ----------------
import os
from typing import List
## ------------------
## --- Third-party---
## ------------------
import numpy as np
import pandas as pd
import torch as t
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
## -----------
## --- Own ---
## -----------
from utils import create_directory


device = t.device('cuda' if t.cuda.is_available() else 'cpu')
class Trainerdense:
    """
    Torch Trainer for model
    """
    def __init__(self,
                 dataset_name: str,
                 model,
                 selected_model: str,
                 loss_func,
                 optim=None,  ## optimizer
                 scheduler=None,
                 train_set=None,
                 test_set=None,
                 norm='L1',
                 cuda=True,
                 early_stopping_cb=None,
                 writer=None,
                 one_matrix=None,
                 criterions: List[str] = ["loss", "accuracy", "f1_score", "precision", "recall"],
                 root_dir: str = None,
                 class_weights=None):
        """
        Args:
            dataset_name: Name of the dataset
            model: Model to be trained
            selected_model: the name of model
            loss_func: Loss Function
            optim: Optimizer
            scheduler: Adaptive Learning Rate for Optimizer
            train_set: Training Dataset
            test_set: Testing Dataset
            norm: Regularization L1 or L2
            cuda: whether to use the GPU
            early_stopping_cb: The stopping criterion
            writer: Tensorboard
            one_matrix: if there is only one input matrix
            criterions: the criterion we want to track, now only [loss, accuracy, f1_score, precision, recall]
            root_dir: the root directory for load data and store checkpoints
            class_weights: sklearn, estimate class weights for unbalanced datasets
        """
        self._dataset_name = dataset_name
        self._model = model
        self._selected_model = selected_model
        self._loss_func = loss_func
        self._optim = optim
        self._scheduler = scheduler
        self._trainset = train_set
        self._testset = test_set
        self._norm = norm
        self._cuda = cuda
        self._early_stopping = early_stopping_cb
        self._writer = writer
        self._one_matrix = one_matrix
        self._root_dir = root_dir
        self._class_weights = None
        if class_weights is not None:
            self._class_weights = t.from_numpy(class_weights)
        if criterions is not None:
            self.criterions = {}
            self.criterions["Dataset"] = self._dataset_name
            self.criterions["Classifier"] = self._selected_model
            for crit in criterions:
                if crit in ["loss"]:
                    self.criterions["train_" + crit] = []
                    self.criterions["val_" + crit] = []
                elif crit in ["f1_score_metric"]: ## use only for test data
                    self.criterions["train_" + crit] = []
                    self.criterions["val_" + crit] = []
                elif crit in ["f1_score"]:
                    self.criterions[crit] = None
                elif crit in ["precision", "recall"]:
                    self.criterions[crit] = 0
                else:
                    raise AttributeError(f"No this criterion {crit}")
        if cuda:
            self._model = self._model.cuda()
            self._loss_func = loss_func.cuda()
        self._num = 0

    def save_checkpoint(self, epoch):
        ckp_directory = self._root_dir + "results/" + self._dataset_name + "/" + \
                                     self._selected_model
        if self._num == 0:
            self._checkpoint_directory = ckp_directory + f"/experiment_{self._num}" + "/checkpoints"
            while os.path.exists(self._checkpoint_directory):
                self._num += 1
                self._checkpoint_directory = ckp_directory + f"/experiment_{self._num}" + "/checkpoints"
            else:
                self._num += 1
        directory_done = create_directory(self._checkpoint_directory)
        print(f"[INFO] the store checkpoint-directory is {self._checkpoint_directory}")
        t.save({'state_dict': self._model.state_dict()},
               self._checkpoint_directory + "/checkpoint_{:03d}.ckp".format(epoch))

    def restore_checkpoint(self, epoch_n):
        model_ckp = t.load(
            self._checkpoint_directory + '/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(model_ckp['state_dict'])

    def train(self, x, y):
        self._optim.zero_grad()
        predicted = self._model(x)
        loss = self._loss_func(predicted,
                               t.tensor(y, dtype=t.long))
        if self._class_weights is not None and len(y[0]) == 1:
            weight_ = self._class_weights[y.view(-1).long()]
            loss = loss * weight_.to(device)
            loss = loss.sum()
        if loss.requires_grad is False:
            loss.requires_grad = True
        loss.backward()
        self._optim.step()
        ## tensorboard for the gradient histogram  --> Todo

        return loss.item(), predicted

    def train_epoch(self):
        self._model.train()
        train_losses = []
        train_f1scores = []
        for i, (data, label) in enumerate(self._trainset, 0):  ## data.size = [batch, dims, len_sample]
            if self._one_matrix:
                xt = data.float().to(device)
                if np.size(label[0].cpu().numpy()) == 1:
                    label = label.to(device).reshape((-1, 1))   ## label should be (len, 1)
                else:
                    label = label.to(device)
                ## Forward pass
                loss, output = self.train(xt, label)
            else:
                NotImplementedError("multiple input not implemented")

            ## calculate the accuracy
            output = t.argmax(output, dim=1)  ## compute the highest probability
            train_losses.append(loss)

            ## for f1 score ## collect all predict and label
            for l, o in zip(label, output):
                f1 = f1_score(l.cpu().detach().numpy(), o.cpu().detach().numpy(), average='macro')
                train_f1scores.append(f1)
        avg_loss = np.mean(train_losses)
        avg_f1 = np.mean(train_f1scores)
        print('[Training] Train Avg. Loss: {}'.format(avg_loss))
        print('-------------------------------------------')
        print('[Training] Train avg F1 rate: {} %'.format(avg_f1))
        print('-------------------------------------------')
        return avg_loss, avg_f1

    def test(self, x, y):
        predicted = self._model(x)
        loss = self._loss_func(predicted,
                               t.tensor(y, dtype=t.long))
        if self._class_weights is not None and len(y[0]) == 1:
            weight_ = self._class_weights[y.view(-1).long()]
            loss = loss * weight_.to(device)
            loss = loss.sum()
        return loss.item(), predicted

    def test_val(self):
        self._model.eval()
        test_losses = []
        test_f1scores = []
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0): ## data.size = [batch, dims, len_sample]
                if self._one_matrix:
                    xt = data.float().to(device)
                    if np.size(label[0].cpu().numpy()) == 1:
                        label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)
                    else:
                        label = label.to(device)
                    ## Forward pass
                    loss, output = self.test(xt, label)

                ## calculate the accuracy
                output = t.argmax(output, dim=1)  ## compute the highest probability
                test_losses.append(loss)

                ## for f1 score ## collect all predict and label
                for l, o in zip(label, output):
                    f1 = f1_score(l.cpu().detach().numpy(), o.cpu().detach().numpy(), average='macro')
                    test_f1scores.append(f1)

        ## f1 score calculation
        avg_f1scores = np.mean(test_f1scores)
        avg_loss = np.mean(test_losses)
        print('[Evaluation] Test Avg. Loss: {}'.format(avg_loss))
        print('-------------------------------------------')
        print('[Evaluation] f1 score', avg_f1scores)

        return avg_loss, avg_f1scores

    def fit(self, epochs=-1):
        assert self._early_stopping is not None or epochs > 0
        self.epoch_count = 1
        best_loss = np.Inf
        best_epoch = None

        while True:
            if self.epoch_count > epochs:
                df_cm = self.evaluate(best_epoch=best_epoch)
                print("[Trainer] Work done!")
                break
            print('[Training] epoch : {}'.format(self.epoch_count))
            train_loss, train_f1 = self.train_epoch()
            test_loss, test_f1 = self.test_val()

            ## scheduler update learning rate
            if self._scheduler is not None:
                self._scheduler.step(test_loss)

            if test_loss < best_loss:   ## loss is the criterion to store the best epoch(model)
                best_epoch = self.epoch_count
                best_loss = test_loss
                self.save_checkpoint(best_epoch)

            ## store loss, accuracy, f1s
            if "train_loss" in self.criterions.keys():
                self.criterions["train_loss"].append(train_loss)
                self.criterions["val_loss"].append(test_loss)

            if "train_f1_score_metric" in self.criterions.keys():
                self.criterions["train_f1_score_metric"].append(train_f1)
                self.criterions["val_f1_score_metric"].append(test_f1)

            ## early stopping
            if self._early_stopping is not None:
                self._early_stopping.step(test_loss)
                self._early_stopping.should_stop()
                if self._early_stopping.early_stop is True:
                    print('[EarlyStopping] The best score is on epoch {} and the loss {}'.format(best_epoch, best_loss))
                    df_cm = self.evaluate(best_epoch=best_epoch)
                    print("[Trainer] Work done! with Early Stopping")
                    break
            ## next epoch
            self.epoch_count += 1
        return self.criterions, df_cm

    def evaluate(self, best_epoch):
        print('[Evaluation] best_epoch for loss: {}'.format(best_epoch))
        print('---- confusion matrix on Validation set for the best epoch Loss ----')
        self.restore_checkpoint(best_epoch)
        self._model.eval()
        f1_scores = []
        labels = None
        predictions = None
        ## f1 score, precision, recall
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0):    ## data.size = [batch, dims, len_sample]
                if self._one_matrix:
                    xt = data.float().to(device)
                    if np.size(label[0].cpu().numpy()) == 1:
                        label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)
                    else:
                        label = label.to(device)
                    ## Forward pass
                    predicted = self._model(xt)

                ## calculate the accuracy
                predicted = t.argmax(predicted, dim=1)  ## compute the highest probability
                for l, p in zip(label, predicted):
                    f1 = f1_score(l.cpu().detach().numpy(), p.cpu().detach().numpy(), average='macro')
                    f1_scores.append(f1)

                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
                if labels is None:
                    labels = label
                else:
                    labels = t.cat((labels, label), dim=0)
        avg_f1 = np.mean(f1_scores)
        print('[Evaluation F1] On testset F1 rate: {} % \n'.format(avg_f1))
        num_cls = len(np.unique(labels.cpu().numpy().reshape(-1, 1)))

        predictions = np.asarray(predictions.cpu().detach().numpy()).reshape(-1, 1)
        labels = np.asarray(labels.cpu().detach().numpy()).reshape(-1, 1)

        ## confusion matrix
        cm = confusion_matrix(labels, predictions, labels=range(num_cls))    ## cm row (true labels) col (predicted labels)
        print(cm)
        df_cm = pd.DataFrame(cm, index=[i for i in range(num_cls)],
                             columns=[i for i in range(num_cls)])
        ## classification report
        val_report = classification_report(labels, predictions, labels=range(num_cls))
        print("Validation report is {}".format(val_report))
        self.criterions["confusion_matrix_valset"] = cm
        self.criterions["best_epoch"] = best_epoch

        return df_cm

    def evaluate_model(self, dataset, best_epoch):
        """
        Get the average accruacy of the dataset on the deep learning Model (best Epoch)
        Returns
        -------
        accuracy: float
        """
        print('[Evaluation] best_epoch for loss: {}'.format(best_epoch))
        self.restore_checkpoint(best_epoch)
        self._model.eval()
        f1_scores = []
        labels = None
        predictions = None
        losses = []
        with t.no_grad():
            for i, (data, label) in enumerate(dataset, 0): ## data size = [B, C(feature), length]
                xt = data.float().to(device)
                if np.size(label[0].cpu().numpy()) == 1:
                    label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)
                else:
                    label = label.to(device)

                ## Forward pass
                predicted = self._model(xt)
                loss = self._loss_func(predicted,
                                       t.tensor(label, dtype=t.long))
                if self._class_weights is not None:
                    weight_ = self._class_weights[label.view(-1).long()]
                    loss = loss * weight_.to(device)
                    loss = loss.sum()

                predicted = t.argmax(predicted, dim=1)
                losses.append(loss.item())
                for l, p in zip(label, predicted):
                    f1 = f1_score(l.cpu().detach().numpy(), p.cpu().detach().numpy(), average='macro')
                    f1_scores.append(f1)
                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
                if labels is None:
                    labels = label
                else:
                    labels = t.cat((labels, label), dim=0)

        avg_loss = np.mean(losses)
        avg_f1 = np.mean(f1_scores)
        print('[Evaluation F1] On testset F1 rate: {} % \n'.format(avg_f1))

        num_cls = len(np.unique(labels.cpu().numpy().reshape(-1, 1)))

        precisions = []
        recalls = []
        f1_s = []
        for i in range(labels.shape[0]):
            l = labels[i].cpu().detach().numpy()
            p = predictions[i].cpu().detach().numpy()
            ## f1 score. precision, recall, best epoch
            if "precision" in self.criterions.keys():
                precisions.append(precision_score(l, p, average='macro'))
            if "recall" in self.criterions.keys():
                recalls.append(recall_score(l, p, average='macro'))
            if "f1_score" in self.criterions.keys():
                f1_s.append(f1_score(l, p, average='macro'))
        if "precision" in self.criterions.keys():
            self.criterions["precision_bestepoch"] = np.mean(precisions)
        if "recall" in self.criterions.keys():
            self.criterions["recall_bestepoch"] = np.mean(recalls)
        if "f1_score" in self.criterions.keys():
            self.criterions["f1_bestepoch"] = np.mean(f1_s)

        if np.size(labels[0].cpu().numpy()) == 1:
            predictions = np.asarray(predictions.cpu().numpy()).reshape(-1, 1)
            labels = np.asarray(labels.cpu().numpy()).reshape(-1, 1)
        else:
            predictions = predictions.cpu().detach().numpy().flatten().tolist()
            labels = labels.cpu().detach().numpy().flatten().tolist()
        ## confusion matrix
        cm = confusion_matrix(labels, predictions,
                              labels=range(num_cls))  ## cm row (true labels) col (predicted labels)
        print('---- confusion matrix on Test set for the best epoch Loss ----')
        print(cm)
        df_cm = pd.DataFrame(cm, index=[i for i in range(num_cls)],
                             columns=[i for i in range(num_cls)])
        ## classification report
        test_report = classification_report(labels, predictions,
                                            labels=range(num_cls))
        print("Test report is {}".format(test_report))

        self.criterions["test_report"] = test_report
        self.criterions["confusion_matrix_testset"] = cm
        self.criterions["testset_f1"] = avg_f1
        self.criterions["testset_loss"] = avg_loss
        return avg_f1, df_cm