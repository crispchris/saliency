"""Pytorch Deep Learning Model trainer"""
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
class Trainer:
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
                 loss_lambda=0.001,
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
            loss_lambda: if regularization is l1 or l2, then we need this
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
        self._loss_lambda = loss_lambda
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
                if crit in ["loss", "accuracy"]:
                    self.criterions["train_" + crit] = []
                    self.criterions["val_" + crit] = []
                elif crit in ["f1_score"]: ## use only for test data
                    self.criterions[crit] = []
                elif crit in ["precision", "recall"]:
                    self.criterions[crit] = 0
                else:
                    raise AttributeError(f"No this criterion {crit}")
        if cuda:
            self._model = self._model.cuda()
            self._loss_func = loss_func.cuda()
        self._num = 0

    def save_checkpoint(self, epoch):
        # ckp_directory = self._root_dir + "results/" + self._dataset_name + "/" + \
        #                              self._selected_model
        ckp_directory = self._root_dir
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
                               t.tensor(y, dtype=t.long).cuda())
        if self._norm == "l1":
            l1_norm = sum(w.abs().sum() for w in self._model.parameters()) * self._loss_lambda
            loss += l1_norm
        elif self._norm == "l2":
            l2_norm = sum(w.pow(2.0).sum() for w in self._model.parameters()) * self._loss_lambda
            loss += l2_norm
        else:
            pass

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
        correct = 0
        sum_labels = 0
        for i, (data, label) in enumerate(self._trainset, 0):  ## data.size = [batch, dims, len_sample]
            if self._one_matrix:
                xt = data.float().to(device)
                label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)
                ## Forward pass
                loss, output = self.train(xt, label)
            else:
                NotImplementedError("multiple input not implemented")

            ## calculate the accuracy
            output = t.argmax(output, dim=1)  ## compute the highest probability
            correct += (output == label).sum().item()
            sum_labels += len(label)
            train_losses.append(loss)
        avg_loss = np.mean(train_losses)
        avg_acc = (correct / sum_labels) * 100
        print('[Training] Train Avg. Loss: {}'.format(avg_loss))
        print('-------------------------------------------')
        print('[Training] Train avg accuracy rate: {} %'.format(avg_acc))
        print('-------------------------------------------')
        return avg_loss, avg_acc

    def test(self, x, y):
        predicted = self._model(x)
        loss = self._loss_func(predicted,
                               t.tensor(y, dtype=t.long).cuda())
        if self._class_weights is not None and len(y[0]) == 1:
            weight_ = self._class_weights[y.view(-1).long()]
            loss = loss * weight_.to(device)
            loss = loss.sum()
        return loss.item(), predicted

    def test_val(self):
        self._model.eval()
        test_losses = []
        ground_true = None
        predicted_label = None
        correct = 0
        sum_labels = 0
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0): ## data.size = [batch, dims, len_sample]
                if self._one_matrix:
                    xt = data.float().to(device)
                    label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)
                    ## Forward pass
                    loss, output = self.test(xt, label)

                ## calculate the accuracy
                output = t.argmax(output, dim=1)  ## compute the highest probability
                correct += (output == label).sum().item()
                sum_labels += len(label)
                test_losses.append(loss)

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
        avg_loss = np.mean(test_losses)
        # avg_loss = t.mean(t.stack(test_loss)).cpu().numpy()
        avg_acc = (correct / sum_labels) * 100
        print('[Evaluation] Test Avg. Loss: {}'.format(avg_loss))
        print('-------------------------------------------')
        print('[Evaluation] Test avg accuracy rate: {} %'.format(avg_acc))
        print('-------------------------------------------')
        print('[Evaluation] f1 score', f1)

        return avg_loss, avg_acc, f1

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
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, f1 = self.test_val()

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
            if "train_accuracy" in self.criterions.keys():
                self.criterions["train_accuracy"].append(train_acc)
                self.criterions["val_accuracy"].append(test_acc)
            if "f1_score" in self.criterions.keys():
                self.criterions["f1_score"].append(f1)

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
        correct = 0
        labels = None
        predictions = None
        ## f1 score, precision, recall
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0):    ## data.size = [batch, dims, len_sample]
                if self._one_matrix:
                    xt = data.float().to(device)
                    label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)
                    ## Forward pass
                    predicted = self._model(xt)

                ## calculate the accuracy
                predicted = t.argmax(predicted, dim=1)  ## compute the highest probability
                correct += (predicted == label).sum().item()
                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
                if labels is None:
                    labels = label
                else:
                    labels = t.cat((labels, label), dim=0)

            sum_labels = len(labels)
        avg_acc = (correct / sum_labels) * 100
        print('[Evaluation Accuracy] On Validation Set accuracy rate: {} % \n'.format(avg_acc))

        num_cls = len(np.unique(labels.cpu().numpy().reshape(-1, 1)))

        predictions = np.asarray(predictions.cpu().numpy()).reshape(-1, 1)
        labels = np.asarray(labels.cpu().numpy()).reshape(-1, 1)

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
        correct = 0
        labels = None
        predictions = None
        losses = []
        with t.no_grad():
            for i, (data, label) in enumerate(dataset, 0): ## data size = [B, C(feature), length]
                xt = data.float().to(device)
                label = label.to(device).reshape((-1, 1))  ## label should be (len, 1)

                ## Forward pass
                predicted = self._model(xt)
                loss = self._loss_func(predicted,
                                       t.tensor(label, dtype=t.long).cuda())
                if self._class_weights is not None:
                    weight_ = self._class_weights[label.view(-1).long()]
                    loss = loss * weight_.to(device)
                    loss = loss.sum()

                predicted = t.argmax(predicted, dim=1)
                correct += (predicted == label).sum().item()
                losses.append(loss.item())
                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
                if labels is None:
                    labels = label
                else:
                    labels = t.cat((labels, label), dim=0)

        avg_loss = np.mean(losses)
        sum_labels = len(labels)
        avg_acc = (correct / sum_labels) * 100
        print('[Evaluation Accuracy] On testset accuracy rate: {} % \n'.format(avg_acc))

        num_cls = len(np.unique(labels.cpu().numpy().reshape(-1, 1)))

        predictions = np.asarray(predictions.cpu().numpy()).reshape(-1, 1)
        labels = np.asarray(labels.cpu().numpy()).reshape(-1, 1)

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

        ## f1 score. precision, recall, best epoch
        if "precision" in self.criterions.keys():
            precision = precision_score(labels, predictions, average='macro')
            self.criterions["precision_bestepoch"] = precision
        if "recall" in self.criterions.keys():
            recall = recall_score(labels, predictions, average='macro')
            self.criterions["recall_bestepoch"] = recall
        if "f1_score" in self.criterions.keys():
            f1_s = f1_score(labels, predictions, average='macro')
            self.criterions["f1_bestepoch"] = f1_s

        self.criterions["test_report"] = test_report
        self.criterions["confusion_matrix_testset"] = cm
        self.criterions["testset_acc"] = avg_acc
        self.criterions["testset_loss"] = avg_loss
        return avg_acc, df_cm


