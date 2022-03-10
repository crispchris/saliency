"""
Metric: Temporal Sequence Evaluation
In Time Series Data, there is dependency inside the time series (inter-dependency)
In Temporal Importance/Insertion-Deletion,
we only evaluate the single feature at one time or not completely continuous sequence.
Here we add Swap and mean time points into our feature baselines.
References: https://arxiv.org/abs/1909.07082
"""

## -------------------
## --- Third-Party ---
## -------------------
import os
import sys
sys.path.append('..')
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
import torch as t
t.cuda.empty_cache()
import torch.nn as nn

class TemporalSequenceEval:
    def __init__(self,
                 model: nn.Module,
                 eval_mode: str,
                 length: float,
                 device=None):
        """
        Create a metric instance to evaluate temporal sequence dependency, created by Saliency Maps
        (Interpretation Methods)
        This can also have two views: Importance Sequence will be taken out
                                        Random Sequence will be taken out
        Compare the Prediction change with the original prediction

        Parameters
        ----------
        model (nn.Module): Deep learning model to be explained
        eval_mode (str): "swap" or "mean" or "zero"
                        "swap": Swap time points, which swaps the time series points
                                Tsub = (ti, ti+1, ..., ti+ns) with length ns
                                Tsub reverse = (ti+ns, ti+ns-1, ... ti) with length ns
                        "mean": Like the same sequence, instead of swaping the sequence,
                                it will be replaced by their mean
                        "zero": Use the same sequence and set them to zeros
        length (float) : ns, the length of the sequence, which should be modified(perturbated)
                        here, the percentage of the length in the whole sample will be used
        device: Torch CPU or GPU
        """
        self.model = model
        self.eval_mode = eval_mode
        self.length = length
        self.device = device

    def single_run(self, sample: np.ndarray,
                   saliency_map: np.ndarray,
                   verbose=0,
                   save_to: str = None):
        """
        Run metric on a single sample, and see how it works

        Parameters
        ----------
        sample (np.ndarray) : normalized sample [D, L]
        saliency_map (np.ndarray) : saliency map from a vis.Method [D, L]
        verbose (int): 0 - No plot (return the scores and the comparison)
                       1 - print the original signal, perturbated signal and the score
        save_to (str) : directory to save plots

        Returns
        -------
        origin_score (float) : the prediction for original sample from model
        modified_score (float) : the prediction for modified sample from model
        gap_scores (float) : the gap score (comparison) between original sample and modified sample
                            (gap_scores = origin_score - modified_score )
                            larger is better (positive)
        """
        ## one single sample
        sample = t.tensor(sample).float().to(self.device)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        ## prediction of original sample
        pred = self.model(sample) ## pred shape [#Batch, #Channel, #Length]
                                  ## Channel is Classes (the output for classification)
        prob, idx = t.max(pred, dim=1)
        idx = idx.cpu().numpy()[0]

        print("[Original sample] The predicted Class {} : Probability {:.4f}".format(idx[0],
                                                                            float(prob[0][0])))
        origin_score = prob[0][0]

        ## find the start time point
        sample_len = saliency_map.shape[1]
        sample_channel = saliency_map.shape[0]
        saliency_order = np.unravel_index(np.flip(np.argsort(saliency_map.ravel())), shape=(
            saliency_map.shape[0],
            saliency_map.shape[-1]))
        # sali_max, max_idx = t.max(t.tensor(saliency_map), dim=-1) ## saliency_map has shape of [#Feature, #Time]
                                                        ## We look the max on the time axis
        ## find also random start time point
        np.random.seed(42)
        random_saliency_order = np.argsort(saliency_map.ravel())
        np.apply_along_axis(np.random.shuffle, axis=-1, arr=random_saliency_order)
        random_saliency_order = np.unravel_index(random_saliency_order, shape=(saliency_map.shape[0],
                                                                               saliency_map.shape[-1]))
        ## Now we have the indexes of the max values in saliency map
        ## set the indexes of the max values as the middle point in the perturbated time points
        ## Unless they are at the beginning or the end of the sample
        whole_len = np.ceil(sample_len * self.length)
        half_len = int((whole_len - 1) / 2)
        min_idx = np.maximum(0, saliency_order[1][0] - half_len)
        max_idx = np.minimum(saliency_order[1][0] + half_len, sample_len)
        ran_min_idx = np.maximum(0, random_saliency_order[1][0] - half_len)
        ran_max_idx = np.maximum(random_saliency_order[1][0] + half_len, sample_len)
        # min_idx = np.maximum([0]*sample_channel, max_idx - half_len)
        # max_idx = np.minimum(max_idx + half_len, [sample_len]*sample_channel)
        # ran_min_idx = np.maximum([0]*sample_channel, random_sali_order[:, 0] - half_len)
        # ran_max_idx = np.minimum(random_sali_order[:, 0] + half_len, [sample_len]*sample_channel)

        ## Copy the sample
        modified_sample = sample.clone().detach()
        ran_modified_sample = sample.clone().detach()

        ## extract the signal
        signal = None
        ran_signal = None
        # for i in range(sample_channel):
        if signal is None:
            signal = sample[0, saliency_order[0][0], min_idx:max_idx+1]
            ran_signal = sample[0, random_saliency_order[0][0], ran_min_idx:ran_max_idx+1]
        # else:
        #     signal = np.concatenate((signal, sample[0, i, min_idx[i]:max_idx[i]+1]), axis=0)
        #     ran_signal = np.concatenate((ran_signal, sample[0, i, ran_min_idx[i]:ran_max_idx[i]+1]),
        #                                 axis=0)
        signal = signal.reshape(-1)
        ran_signal = ran_signal.reshape(-1)
        ## evaluation mode to be selected
        if self.eval_mode in ["swap"]:
            signal_ = t.flip(signal, dims=[-1])
            ran_signal_ = t.flip(ran_signal, dims=[-1])
        elif self.eval_mode in ["mean"]:
            signal_ = t.mean(signal, dim=-1)
            ran_signal_ = t.mean(ran_signal, dim=-1)
        elif self.eval_mode in ["zero"]:
            signal_ = t.zeros(signal.shape)
            ran_signal_ = t.zeros(ran_signal.shape)
        else:
            raise ValueError("evaluation mode is wrong, it should be either swap or mean or zero")
        # for i in range(sample_channel):
        modified_sample[0, saliency_order[0][0], min_idx:max_idx+1] = signal_
        ran_modified_sample[0, random_saliency_order[0][0], ran_min_idx:ran_max_idx+1] = ran_signal_

        ## prediction of modified model
        modified_pred = self.model(modified_sample)
        mod_prob = modified_pred[0, idx[0], 0] ## probability for the same class
        modified_ran_pred = self.model(ran_modified_sample)
        mod_ran_prob = modified_ran_pred[0, idx[0], 0]

        print("[Modified sample] The predicted Class {} : Probability {:.4f}".format(idx[0],
                                                                            float(mod_prob)))
        print("[Modified random sample] The predicted Class {} : Probability {:.4f}".format(idx[0],
                                                                            float(mod_ran_prob)))
        modified_score = mod_prob
        modified_ran_score = mod_ran_prob
        gap_scores = origin_score - modified_score
        gap_ran_scores = origin_score - modified_ran_score

        ## plot
        if verbose == 1:
            color_mapping = plt.get_cmap().colors
            cmap = color_mapping[0::256 // modified_sample.shape[1]]

            plt.figure(figsize=(10, 5))
            plt.subplot(131)
            ylabel0 = 'Original sample in time domain'
            plt.title('{} \n Prob = {:.4f}'.format(ylabel0,
                                                   origin_score))
            for dim in range(sample.shape[1]):
                plt.plot(np.arange(sample.shape[-1]),
                         sample.cpu().detach().numpy()[0, dim, :],
                         c= cmap[dim])
            plt.xlabel("time steps")
            plt.ylabel("[Original] Sensors values")

            plt.subplot(132)
            ylabel2 = f'Importance sequence in Time Domain {self.eval_mode}, percent: {self.length}'
            plt.title('{} , Prob = {:.4f} \n Gap with origin = {:.4f}'.format(ylabel2,
                                                                              modified_score,
                                                                              gap_scores))
            for dim in range(modified_sample.shape[1]):
                plt.plot(np.arange(modified_sample.shape[-1]),
                         modified_sample.cpu().detach().numpy()[0, dim, :],
                         c= cmap[dim])
            plt.xlabel("time steps")
            plt.ylabel(f"[Perturbate {self.eval_mode}] Sensors values")

            plt.subplot(133)
            ylabel2 = f'Random sequence in Time Domain {self.eval_mode}, percent: {self.length}'
            plt.title('{} , Prob = {:.4f} \n Gap with origin = {:.4f}'.format(ylabel2,
                                                                              modified_ran_score,
                                                                              gap_ran_scores))
            for dim in range(ran_modified_sample.shape[1]):
                plt.plot(np.arange(ran_modified_sample.shape[-1]),
                         ran_modified_sample.cpu().detach().numpy()[0, dim, :],
                         c=cmap[dim])
            plt.xlabel("time steps")
            plt.ylabel(f"[Random {self.eval_mode}] Sensors values")
            if save_to is not None:
                plt.tight_layout()
                plt.savefig(save_to + f'/time_sequence_eval_{self.eval_mode}_len_{self.length}.png')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
        return gap_scores, gap_ran_scores, modified_sample


    def evaluation(self, batch_samples: np.ndarray,
                   batch_labels: List,
                   batch_saliency_maps: np.ndarray,
                   batch_size: int,
                   verbose: int = 0,
                   method: str = None,
                   typeofSaliency: str = None,
                   save_to: str = None):
        """
        Evaluation an explanation on a dataset and return the average scores

        Parameters
        ----------
        batch_samples (np.ndarray): batch of samples [Batch, Features(dim), Length]
        batch_labels (List): batch of Labels (corresponding to batch_samples)
        batch_saliency_maps (np.ndarray): batch of explanations [Batch, Features(dim), Length]
        batch_size (int) : number of sample for one batch
        verbose (int): 0 - No plot (return the scores and the comparison)
                       1 - print the gap scores through the dataset
        method (str) : only use for the save name, method from vis. Method
        typeofSaliency (str): which Saliency is used, either No_abs or Abs norm
        save_to (str) : directory to save plots

        Returns
        -------
        importance_gap_scores (float): the average gap score for the whole dataset, when the time points
                                        are picked as saliency-maps identify them as importance
                                        size: [#Batch]
        random_gap_scores (float) : the average gap score for the whole dataset, when the time points
                                        are randomly picked
                                        size: [#Batch]

        """
        num_samples = batch_samples.shape[0]
        num_classes = len(np.unique(batch_labels))
        print(f"batch labels {np.unique(batch_labels)}")
        assert num_samples % batch_size == 0

        predictions = np.zeros((num_samples, num_classes))
        #predictions = t.FloatTensor(num_samples, num_classes)
        batch_samples = t.tensor(batch_samples).float().to(self.device)
        for i in tqdm(range(num_samples // batch_size), desc="Predicting Labels"):
            t.cuda.empty_cache()
            preds = self.model(batch_samples[i * batch_size:(i + 1) * batch_size])
            predictions[i * batch_size:(i + 1) * batch_size] = preds.squeeze(dim=-1).cpu().detach().numpy()
            del preds
        ## the prediction classes of the samples
        top = np.argmax(predictions, axis=1)

        ## find the start time point
        sample_len = batch_samples.shape[-1]
        sample_channel = batch_samples.shape[1]
        saliency_orders = (np.zeros((batch_saliency_maps.shape[0],
                           batch_saliency_maps.shape[1] * batch_saliency_maps.shape[-1])),
                           np.zeros((batch_saliency_maps.shape[0],
                           batch_saliency_maps.shape[1] * batch_saliency_maps.shape[-1])))
        random_saliency_orders = deepcopy(saliency_orders)
        for i in range(batch_samples.shape[0]):
            order = np.flip(np.argsort(batch_saliency_maps[i].ravel()))
            saliency_orders[0][i], saliency_orders[1][i] = np.unravel_index(order,
                                                 shape=(batch_saliency_maps[i].shape[0],
                                                        batch_saliency_maps[i].shape[-1]))

            ## find also random start time point
            np.random.seed(42)
            # random_saliency_orders = np.argsort(batch_saliency_maps, axis=-1)
            order_from_samples = np.argsort(batch_samples[i].cpu().detach().numpy().ravel())
            np.apply_along_axis(np.random.shuffle, axis=-1, arr=order_from_samples)
            random_saliency_orders[0][i], random_saliency_orders[1][i] = np.unravel_index(order_from_samples,
                                                                          shape=(batch_samples[i].shape[0],
                                                                                 batch_samples[i].shape[-1]))
        #sali_max, max_idx = t.max(t.tensor(batch_saliency_maps), dim=-1)  ## saliency_map has shape of [#Batch, #Feature, #Time]
        ## max_idx should now have the shape as [#Batch, #Feature]
        ## We look the max on the time axis


        ## Now we have the indexes of the max values in saliency map
        ## set the indexes of the max values as the middle point in the perturbated time points
        ## Unless they are at the beginning or the end of the sample
        whole_len = np.ceil(sample_len * self.length)
        half_len = int((whole_len - 1) / 2)
        zeros = np.zeros(batch_samples.shape[0])
        fulls = np.zeros(batch_samples.shape[0]) + sample_len
        min_idx = np.maximum(zeros, saliency_orders[1][:, 0] - half_len).astype("int64")
        max_idx = np.minimum(saliency_orders[1][:, 0] + half_len, fulls).astype("int64")
        ran_min_idx = np.maximum(zeros, random_saliency_orders[1][:, 0] - half_len).astype("int64")
        ran_max_idx = np.minimum(random_saliency_orders[1][:, 0] + half_len, fulls).astype("int64")

        ## Copy the sample
        modified_batch_samples = batch_samples.clone().detach()
        ran_modified_batch_samples = batch_samples.clone().detach()

        ## extract the signals
        ## evaluation mode to be selected
        ## coordination for the saliency maps
        importance_coords_rows = saliency_orders[0][:, 0].astype("int64")
        random_coords_rows = random_saliency_orders[0][:, 0].astype("int64")

        for i in range(batch_samples.shape[0]):
            # for j in range(sample_channel):
            signal = batch_samples[i, importance_coords_rows[i],
                                   slice(min_idx[i], max_idx[i]+1)]
            ran_signal = batch_samples[i, random_coords_rows[i],
                                   ran_min_idx[i]:ran_max_idx[i]+1]
            if self.eval_mode in ["swap"]:
                modified_batch_samples[i, importance_coords_rows[i], min_idx[i]:max_idx[i]+1] = t.flip(
                    signal, dims=[-1]
                )
                ran_modified_batch_samples[i, random_coords_rows[i], ran_min_idx[i]:ran_max_idx[i]+1] = t.flip(
                    ran_signal, dims=[-1]
                )
            elif self.eval_mode in ["mean"]:
                modified_batch_samples[i, importance_coords_rows[i], min_idx[i]:max_idx[i]+1] = t.mean(
                    signal, dim=-1
                )
                ran_modified_batch_samples[i, random_coords_rows[i], ran_min_idx[i]:ran_max_idx[i]+1] = t.mean(
                    ran_signal, dim=-1
                )
            elif self.eval_mode in ["zero"]:
                modified_batch_samples[i, importance_coords_rows[i], min_idx[i]:max_idx[i]+1] = t.zeros(
                    signal.shape
                )
                ran_modified_batch_samples[i, random_coords_rows[i], ran_min_idx[i]:ran_max_idx[i]+1] = t.zeros(
                    ran_signal.shape
                )
            else:
                raise ValueError("evaluation mode is wrong, it should be either swap or mean or zero")

        ## prediction of modified model
#         modified_predictions = t.FloatTensor(num_samples, num_classes)
#         ran_modified_predictions = t.FloatTensor(num_samples, num_classes)
        modified_predictions = np.zeros((num_samples, num_classes))
        ran_modified_predictions = np.zeros((num_samples, num_classes))

        for i in tqdm(range(num_samples // batch_size), desc="Predicting modified samples"):
            t.cuda.empty_cache()
            modified_preds = self.model(modified_batch_samples[i * batch_size:(i + 1) * batch_size])
            ran_modified_preds = self.model(ran_modified_batch_samples[i * batch_size:(i + 1) * batch_size])
            modified_predictions[i * batch_size:(i + 1) * batch_size] = modified_preds.squeeze(dim=-1).cpu().detach().numpy()
            ran_modified_predictions[i * batch_size:(i + 1) * batch_size] = ran_modified_preds.squeeze(dim=-1).cpu().detach().numpy()
            del modified_preds
            del ran_modified_preds

        row = np.arange(num_samples)
        modified_score = modified_predictions[row, top]
        modified_ran_score = ran_modified_predictions[row, top]
        gap_scores = predictions[row, top] - modified_score
        gap_ran_scores = predictions[row, top] - modified_ran_score

        # gap_mean_scores = t.mean(gap_scores).cpu().detach().numpy()
        # gap_ran_mean_scores = t.mean(gap_ran_scores).cpu().detach().numpy()

        ## plot
        if verbose == 1:
            color_mapping = plt.get_cmap().colors
            cmap = color_mapping[0::256 // modified_batch_samples.shape[1]]

            plt.figure(figsize=(12, 7))
#             plt.suptitle(f'[Evaluation on {method} with Saliency {typeofSaliency}] Temporal Continuous Sequence {self.eval_mode}ed and percent: {self.length}')
            plt.suptitle(f'[Evaluation on {method}] Temporal Continuous Sequence {self.eval_mode}ed and percent: {self.length}')
            plt.subplot(131)
            ylabel0 = '[Example] An original sample'
            plt.title('{} \n Prob = {:.4f}'.format(ylabel0,
                                                  predictions[0,top[0]]))
            for dim in range(batch_samples.shape[1]):
                plt.plot(range(batch_samples.shape[-1]),
                         batch_samples.cpu().detach().numpy()[0, dim, :],
                         c=cmap[dim])
            plt.xlabel("time steps")
            plt.ylabel("[Original] Sensors values")

            plt.subplot(132)
            ylabel2 = f'[Example] Importance sequence, percent: {self.length}'
            plt.title('{} , \n Prob = {:.4f} \n Gap with origin = {:.4f}'.format(ylabel2,
                                                                              modified_score[0],
                                                                              gap_scores[0]))
            for dim in range(modified_batch_samples.shape[1]):
                plt.plot(range(modified_batch_samples.shape[-1]),
                         modified_batch_samples.cpu().detach().numpy()[0, dim, :],
                         c=cmap[dim])
            plt.xlabel("time steps")
            plt.ylabel(f"[Perturbate {self.eval_mode}] Sensors values")

            plt.subplot(133)
            ylabel2 = f'The gap scores through the whole dataset'
            plt.title('{}'.format(ylabel2))
            plt.plot(np.arange(len(gap_scores)),
                     gap_scores,
                     label = 'importances')
            plt.plot(np.arange(len(gap_ran_scores)),
                     gap_ran_scores,
                     label = 'ranodms')
            plt.xlabel("Number of Sample")
            plt.ylabel(f"[GAP] Scores between Origin - Modified ")
            plt.legend()

            if save_to is not None:
                import os
                print("current working directory is", os.getcwd())
                try:
                    os.mkdir(save_to)
                except OSError as error:
                    print(error)
                plt.tight_layout()
#                 plt.savefig(save_to + f'/{method}_time_sequence_eval_{self.eval_mode}_len_{self.length}_with_sali_{typeofSaliency}_wholedata.png')
                plt.savefig(save_to + f'/{method}_time_sequence_eval_{self.eval_mode}_len_{self.length}_wholedata.png')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()

        return gap_scores, gap_ran_scores
