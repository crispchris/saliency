"""
Metric: inspired by Insertion/Deletion
Compare the importance of temporal feature with the unimportance feature
"""

## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import auc as sk_auc
from tqdm import tqdm
from typing import List
import torch as t
import torch.nn as nn
from copy import deepcopy


class TemporalImportance:
    def __init__(self, model: nn.Module,
                 substrate_fn,
                 eval_mode: str = "mean",
                 step: int = 1,
                 device=None):
        """
            Create a metric instance to evaluate temporal importance
            three steps: take out the non-importance saliency,
                       take out the importance saliency
                       take out with random order
                       Compare the difference

            Parameters
            ----------
            model (nn.Module): Deep learning model to be explained
            step (int) : number of pixels modified per one iteration
                        default: 1 (one pixel per step)
            substrate_fn (func) : a mapping from old pixes to new pixels
            eval_mode (str) : Give a description of the substrate_fn
            device: torch CPU or GPU
        """
        self.model = model
        self.step = step
        self.substrate_fn = substrate_fn
        self.eval_mode = eval_mode
        self.device = device

    def single_run(self, sample: np.ndarray,
                   saliency_map: np.ndarray,
                   percent: float = 0.3,
                   verbose=0,
                   save_to=None):
        """
        Run metric on a single sample

        Parameters
        ----------
        sample (np.ndarray) : normalized sample [D, L]
        saliency_map (np.ndarray) : saliency map from a vis.Method [D, L]
        percent (float) : how many percent of features should be modified,
                        default: 0.3

        verbose (int) : in [0, 1, 2].
            0 - return list of scores (no plot)
            1 - print the top 2 classes
            2 - plot every step
        save_to (str) : directory to save plots

        Returns
        -------
        importance_scores (np.ndarray) : Array containing scores at every step
        unimportance_scores (np.ndarray) : Array containing scores at every step
        """
        sample = t.tensor(sample).float().to(self.device)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])

        ## prediction of original sample
        pred = self.model(sample)  ## pred shape [B, C, L] C: Classes or Features
        prob, idx = t.max(pred, dim=1)
        idx = idx.cpu().numpy()[0] ## predicted class

        if verbose == 2:
            print("The predicted Class {} : Probability {:.4f}".format(idx[0],
                                                                       float(prob[0][0])))
        num_features_tobemodified = np.ceil(sample.shape[1] * sample.shape[-1] * percent)
        assert num_features_tobemodified % self.step == 0
        ## number of steps
        num_steps = int(num_features_tobemodified // self.step)

        ## start and finish arrays
        print("[Metric] Create start and finish sample for both (un/importance)")
        importancestart_sample = sample.clone().detach()
        importancefinish_sample = self.substrate_fn(sample)
        unimportancestart_sample = sample.clone().detach()
        unimportancefinish_sample = self.substrate_fn(sample)
        randomstart_sample = sample.clone().detach()
        randomfinish_sample = self.substrate_fn(sample)
        assert importancestart_sample is not None
        assert importancefinish_sample is not None
        assert unimportancestart_sample is not None
        assert unimportancefinish_sample is not None
        assert randomstart_sample is not None
        assert randomfinish_sample is not None

        importance_scores = np.empty(num_steps+1)  ## include first step is non-change
        unimportance_scores = np.empty(num_steps+1) ## include first step is non-change
        random_scores = np.empty(num_steps + 1) ## include first step is non-change

        ## Coordinates of features(pixels) in order of decreasing saliency
        # saliency_map = np.expand_dims(saliency_map, axis=0)
        ## un/importance order and random order
        ## shape [#feature, #Time step, #2]
        un_saliency_order = np.unravel_index(np.argsort(saliency_map.ravel()), shape=(saliency_map.shape[0],
                                                                                      saliency_map.shape[-1]))
        saliency_order = np.unravel_index(np.flip(np.argsort(saliency_map.ravel())), shape=(
            saliency_map.shape[0],
            saliency_map.shape[-1]))
        np.random.seed(42)
        random_saliency_order = np.argsort(saliency_map.ravel())
        np.apply_along_axis(np.random.shuffle, axis=-1, arr=random_saliency_order)
        random_saliency_order = np.unravel_index(random_saliency_order, shape=(saliency_map.shape[0],
                                                                               saliency_map.shape[-1]))

        ## Coloring dimensions
        #color_mapping = plt.get_cmap().colors
        color_mapping = [plt.get_cmap()]
        cmap = color_mapping[0::256 // sample.shape[1]]

        for i in range(num_steps + 1):
            importance_pred = self.model(t.tensor(importancestart_sample).to(self.device))
            importance_prob1, importance_idx1 = t.topk(importance_pred, k=2, dim=1)
            unimportance_pred = self.model(t.tensor(unimportancestart_sample).to(self.device))
            unimportance_prob1, unimportance_idx1 = t.topk(unimportance_pred, k=2, dim=1)
            random_pred = self.model(t.tensor(randomstart_sample).to(self.device))
            random_prob1, random_idx1 = t.topk(random_pred, k=2, dim=1)

            if verbose == 2:
                print("[Importance Scores] Prediction")
                print("(high) First Class {} : Probability {:.4f}".format(
                    importance_idx1[0][0][0], float(importance_prob1[0][0][0])))
                print("Second Class {} : Probability {:.4f}".format(
                    importance_idx1[0][1][0], float(importance_prob1[0][1][0])))
                print("[Unimportance Scores] Prediction")
                print("(high) First Class {} : Probability {:.4f}".format(
                    unimportance_idx1[0][0][0], float(unimportance_prob1[0][0][0])))
                print("Second Class {} : Probability {:.4f}".format(
                    unimportance_idx1[0][1][0], float(unimportance_prob1[0][1][0])))
                print("[Random baseline Scores] Prediction")
                print("(high) First Class {} : Probability {:.4f}".format(
                    random_idx1[0][0][0], float(random_prob1[0][0][0])))
                print("Second Class {} : Probability {:.4f}".format(
                    random_idx1[0][1][0], float(random_prob1[0][1][0])))

            importance_scores[i] = importance_pred[0, idx, 0]
            unimportance_scores[i] = unimportance_pred[0, idx, 0]
            random_scores[i] = random_pred[0, idx, 0]

            if verbose == 2 or (i == num_steps and verbose == 1):
                if verbose == 2 and i == 0:
                    plt.figure(figsize=(10, 5))
                elif verbose == 1:
                    plt.figure(figsize=(10, 5))

                plt.subplot(321)
                ylabel0 = f"Importance in Time Domain (Delete), percent: {percent}"
                plt.title('{} , Step: {:.1f}%, \n Prob={:.4f}'.format(ylabel0,
                                                           100 * i / num_steps,
                                                           importance_scores[i]))
                if t.is_tensor(importancestart_sample):
                    importancestart_sample = importancestart_sample.cpu().numpy()
                for dim in range(importancestart_sample.shape[1]):
                    plt.plot(range(importancestart_sample.shape[-1]),
                             importancestart_sample[0, dim, :],
                             c=cmap[dim])
                plt.xlabel("time steps")
                plt.ylabel("[Perturbated] Sensors values")

                plt.subplot(322)
                title0 = "Area under Curve for importance deletion"
                plt.plot(np.arange(i + 1) / num_steps, importance_scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / num_steps, 0, importance_scores[:i + 1],
                                 alpha=0.4)
                plt.title(title0)
                plt.xlabel(f"Importance Deletion steps from 0% to {percent}%")
                plt.ylabel("Probability for Class {}".format(idx[0]))

                plt.subplot(323)
                ylabel1 = f"Unimportance in Time Domain (Delete), percent: {percent}"
                plt.title('{} {:.1f}%, \n Prob={:.4f}'.format(ylabel1,
                                                           100 * i / num_steps,
                                                           unimportance_scores[i]))
                if t.is_tensor(unimportancestart_sample):
                    unimportancestart_sample = unimportancestart_sample.cpu().numpy()
                for dim in range(unimportancestart_sample.shape[1]):
                    plt.plot(range(unimportancestart_sample.shape[-1]),
                             unimportancestart_sample[0, dim, :],
                             c=cmap[dim])
                plt.xlabel("time steps")
                plt.ylabel("[Perturbated] Sensors values")

                plt.subplot(324)
                title1 = "Area under Curve for Unimportance deletion"
                plt.plot(np.arange(i + 1) / num_steps, unimportance_scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / num_steps, 0, unimportance_scores[:i + 1],
                                 alpha=0.4)
                plt.title(title1)
                plt.xlabel(f"Unimportance Deletion steps from 0% to {percent}%")
                plt.ylabel("Probability for Class {}".format(idx[0]))

                plt.subplot(325)
                ylabel2 = f"Random baseline in Time Domain (Delete), percent: {percent}"
                plt.title('{} {:.1f}%, \n Prob={:.4f}'.format(ylabel2,
                                                              100 * i / num_steps,
                                                              random_scores[i]))
                if t.is_tensor(randomstart_sample):
                    randomstart_sample = randomstart_sample.cpu().numpy()
                for dim in range(randomstart_sample.shape[1]):
                    plt.plot(range(randomstart_sample.shape[-1]),
                             randomstart_sample[0, dim, :],
                             c=cmap[dim])
                plt.xlabel("time steps")
                plt.ylabel("[Perturbated] Sensors values")

                plt.subplot(326)
                title2 = "Area under Curve for Random baseline"
                plt.plot(np.arange(i + 1) / num_steps, random_scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / num_steps, 0, random_scores[:i + 1],
                                 alpha=0.4)
                plt.title(title2)
                plt.xlabel(f"Random Deletion steps from 0% to {percent}%")
                plt.ylabel("Probability for Class {}".format(idx[0]))
                ## save plot
                if save_to is not None and i == num_steps:
                    plt.tight_layout()
                    plt.savefig(save_to + '/importance_deletion_game.png')
                    plt.close()
                elif save_to is not None:
                    plt.tight_layout()
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                elif save_to is None and i == num_steps:
                    plt.tight_layout()
                    plt.show()

            if i < num_steps:
                importance_coords_rows = saliency_order[0][i * self.step:(i + 1) * self.step]
                importance_coords_cols = saliency_order[1][i * self.step:(i + 1) * self.step]
                unimportance_coords_rows = un_saliency_order[0][i * self.step:(i + 1) * self.step]
                unimportance_coords_cols = un_saliency_order[1][i * self.step:(i + 1) * self.step]
                random_coords_rows = random_saliency_order[0][i * self.step:(i + 1) * self.step]
                random_coords_cols = random_saliency_order[1][i * self.step:(i + 1) * self.step]

                ## importance
                # x = np.linspace(0, 1, importance_coords.shape[2]).astype("int32")
                # y = np.linspace(0, importance_coords.shape[1] - 1, importance_coords.shape[1]).astype("int32")
                # xv, yv = np.meshgrid(x, y)
                if t.is_tensor(importancestart_sample):
                    importancestart_sample = importancestart_sample.cpu().numpy()
                importancestart_sample[0][importance_coords_rows, importance_coords_cols] = importancefinish_sample.cpu().numpy()[0][importance_coords_rows,
                                                                   importance_coords_cols]

                ## Unimportance
                # x = np.linspace(0, 1, unimportance_coords.shape[2]).astype("int32")
                # y = np.linspace(0, unimportance_coords.shape[1] - 1, unimportance_coords.shape[1]).astype("int32")
                # xv, yv = np.meshgrid(x, y)
                if t.is_tensor(unimportancestart_sample):
                    unimportancestart_sample = unimportancestart_sample.cpu().numpy()
                unimportancestart_sample[0][unimportance_coords_rows, unimportance_coords_cols] = unimportancefinish_sample.cpu().numpy()[0][unimportance_coords_rows,
                                                                       unimportance_coords_cols]

                ## random baseline
                # x = np.linspace(0, 1, random_coords.shape[2]).astype("int32")
                # y = np.linspace(0, random_coords.shape[1] - 1, random_coords.shape[1]).astype("int32")
                # xv, yv = np.meshgrid(x, y)
                ## Unimportance
                if t.is_tensor(randomstart_sample):
                    randomstart_sample = randomstart_sample.cpu().numpy()
                randomstart_sample[0][random_coords_rows, random_coords_cols] = randomfinish_sample.cpu().numpy()[0][random_coords_rows,
                                                                       random_coords_cols]


        return importance_scores, unimportance_scores, random_scores, (importancestart_sample, unimportancestart_sample)

    def evaluation(self, batch_samples: np.ndarray,
                   batch_labels: List,
                   batch_saliency_maps: np.ndarray,
                   batch_size: int,
                   percent: float = 0.3,
                   method: str = None,
                   verbose: int = 0,
#                    typeofsaliency: str = None,
                   save_to: str = None,
                   random_baseline: bool = False):
        """
        Evaluation an explanation on a dataset and return the average scores

        Parameters
        ----------
        batch_samples (np.ndarray): batch of samples [Batch, Features(dim), Length]
        batch_labels (List): batch of Labels (corresponding to batch_samples)
        batch_saliency_maps (np.ndarray): batch of explanations [Batch, Features(dim), Length]
        batch_size (int): number of sample for one batch
        percent (float) : how many percent of features should be modified,
                        default: 0.3
        method (str) : which method is used now (just use for plot)
        verbose (int) : 0 - No plot (return the scores)
                       1 - print the area under curve of prediction from model,
                            according to (un)importance/random features(Saliency maps)
#        typeofsaliency (str): which type of Saliency is used, No abs or with Abs
        save_to (str) : directory to save the plot
        random_baseline (bool): Add random baseline if True (default: False)

        Returns
        -------
        importance_auc (np.ndarrary) : Array containing (Area under curve) at every sample [Batch, 1]
        unimportance_auc (np.ndarrary) : Array containing (Area under curve) at every sample [Batch, 1]
        if random_baseline:
            random_auc (np.ndarrary) : Array containing (Area under curve) at every sample [Batch, 1]

        importance_scores (np.ndarray) : Array containing scores at every sample [Steps, Batch]
        unimportance_scores (np.ndarray) : Array containing scores at every sample [Steps, Batch]
        if random_baseline:
            random_scores (np.ndarray) : Array containing scores at every sample [Steps, Batch]
        """
        num_samples = batch_samples.shape[0]
        num_classes = len(np.unique(batch_labels))
        assert num_samples % batch_size == 0
        
        predictions = np.zeros((num_samples, num_classes))
        #predictions = t.FloatTensor(num_samples, num_classes)

        batch_samples = t.tensor(batch_samples).float().to(self.device)
        for i in tqdm(range(num_samples // batch_size), desc="Predicting Labels"):
            t.cuda.empty_cache()
            preds = self.model(batch_samples[i * batch_size:(i + 1) * batch_size])
            predictions[i * batch_size:(i + 1) * batch_size] = preds.squeeze(dim=-1).cpu().detach().numpy()
            del preds
        top = np.argmax(predictions, axis=1)

        num_features_tobemodified = np.ceil(batch_samples.shape[1] * batch_samples.shape[-1] * percent)
        assert num_features_tobemodified % self.step == 0
        ## number of steps
        num_steps = int(num_features_tobemodified // self.step)

        importance_scores = np.empty((num_steps + 1, num_samples))  ## include first step is non-change
        unimportance_scores = np.empty((num_steps + 1, num_samples))  ## include first step is non-change
        if random_baseline:
            random_scores = np.empty((num_steps + 1, num_samples))  ## include first step is non-change

        ## un/importance order and random order
        un_saliency_orders = (np.zeros((batch_saliency_maps.shape[0],
                                        batch_saliency_maps.shape[1] * batch_saliency_maps.shape[-1])),
                              np.zeros((batch_saliency_maps.shape[0],
                                        batch_saliency_maps.shape[1] * batch_saliency_maps.shape[-1])))
        saliency_orders = deepcopy(un_saliency_orders)
        if random_baseline:
            random_saliency_orders = deepcopy(un_saliency_orders)
        for i in range(batch_samples.shape[0]):
            order = np.argsort(batch_saliency_maps[i].ravel())
            un_saliency_orders[0][i], un_saliency_orders[1][i] = np.unravel_index(order,
                                                 shape=(batch_saliency_maps[i].shape[0],
                                                        batch_saliency_maps[i].shape[-1]))
            saliency_orders[0][i], saliency_orders[1][i] = np.unravel_index(np.flip(order),
                                               shape=(batch_saliency_maps[i].shape[0],
                                                      batch_saliency_maps[i].shape[-1]))
            if random_baseline:
                np.random.seed(42)
                # random_saliency_orders = np.argsort(batch_saliency_maps, axis=-1)
                order_from_samples = np.argsort(batch_samples[i].cpu().detach().numpy().ravel())
                np.apply_along_axis(np.random.shuffle, axis=-1, arr=order_from_samples)
                random_saliency_orders[0][i], random_saliency_orders[1][i] = np.unravel_index(order_from_samples,
                                                            shape=(batch_samples[i].shape[0],
                                                                   batch_samples[i].shape[-1]))

        ## start and finish arrays
        print("[Metric] Create start and finish sample for both (un/importance)")
        importancestart_sample = batch_samples.clone().detach()
        importancefinish_sample = self.substrate_fn(batch_samples)
        unimportancestart_sample = batch_samples.clone().detach()
        unimportancefinish_sample = self.substrate_fn(batch_samples)
        if random_baseline:
            randomstart_sample = batch_samples.clone().detach()
            randomfinish_sample = self.substrate_fn(batch_samples)
            assert randomstart_sample is not None
            assert randomfinish_sample is not None

        assert importancestart_sample is not None
        assert importancefinish_sample is not None
        assert unimportancestart_sample is not None
        assert unimportancefinish_sample is not None

        caption = "delete"
        for i in tqdm(range(num_steps+1), desc= caption + " features in Length"):
            # Iterate over batches
            for j in range(num_samples // batch_size):
                # Compute prediction for new samples (adjusted)
                importance_preds = self.model(t.tensor(importancestart_sample[j*batch_size:(j+1)*batch_size]).to(self.device))
                unimportance_preds = self.model(t.tensor(unimportancestart_sample[j*batch_size:(j+1)*batch_size]).to(self.device))
                if random_baseline:
                    random_preds = self.model(t.tensor(randomstart_sample[j*batch_size:(j+1)*batch_size]).to(self.device))

                importance_preds = importance_preds.detach().cpu().numpy()[range(batch_size),
                                                                           top[j*batch_size:(j+1)*batch_size]]
                unimportance_preds = unimportance_preds.detach().cpu().numpy()[range(batch_size),
                                                                               top[j * batch_size:(j + 1) * batch_size]]
                if random_baseline:
                    random_preds = random_preds.detach().cpu().numpy()[range(batch_size),
                                                                       top[j * batch_size:(j + 1) * batch_size]]

                importance_scores[i, j*batch_size:(j+1)*batch_size] = importance_preds.reshape(-1)
                unimportance_scores[i, j*batch_size:(j+1)*batch_size] = unimportance_preds.reshape(-1)
                if random_baseline:
                    random_scores[i, j*batch_size:(j+1)*batch_size] = random_preds.reshape(-1)

            if i < num_steps:
                ## find the coordination for the saliency maps
                importance_coords_rows = saliency_orders[0][:, i * self.step:(i + 1) * self.step].astype('int64').reshape(-1)
                importance_coords_cols = saliency_orders[1][:, i * self.step:(i + 1) * self.step].astype('int64').reshape(-1)
                unimportance_coords_rows = un_saliency_orders[0][:, i * self.step:(i + 1) * self.step].astype('int64').reshape(-1)
                unimportance_coords_cols = un_saliency_orders[1][:, i * self.step:(i + 1) * self.step].astype('int64').reshape(-1)
                if random_baseline:
                    random_coords_rows = random_saliency_orders[0][:, i * self.step:(i + 1) * self.step].astype('int64').reshape(-1)
                    random_coords_cols = random_saliency_orders[1][:, i * self.step:(i + 1) * self.step].astype('int64').reshape(-1)

                ## importance
                # x = np.linspace(0, 1, importance_coords.shape[2]).astype("int32")
                # y = np.linspace(0, importance_coords.shape[1] - 1, importance_coords.shape[1]).astype("int32")
                # xv, yv = np.meshgrid(x, y)
                if isinstance(importancestart_sample, t.Tensor):
                    importancestart_sample = importancestart_sample.cpu().numpy()
                importancestart_sample[range(num_samples), importance_coords_rows, importance_coords_cols] = importancefinish_sample.cpu().numpy()[
                                                                                            range(num_samples),
                                                                                            importance_coords_rows,
                                                                                            importance_coords_cols]
                ## Unimportance
                # x = np.linspace(0, 1, unimportance_coords.shape[2]).astype("int32")
                # y = np.linspace(0, unimportance_coords.shape[1] - 1, unimportance_coords.shape[1]).astype("int32")
                # xv, yv = np.meshgrid(x, y)
                if t.is_tensor(unimportancestart_sample):
                    unimportancestart_sample = unimportancestart_sample.cpu().numpy()
                unimportancestart_sample[range(num_samples), unimportance_coords_rows, unimportance_coords_cols] = unimportancefinish_sample.cpu().numpy()[
                                                                                                  range(num_samples),
                                                                                                  importance_coords_rows,
                                                                                                  importance_coords_cols]
                ## Random baseline
                # x = np.linspace(0, 1, random_coords.shape[2]).astype("int32")
                # y = np.linspace(0, random_coords.shape[1] - 1, random_coords.shape[1]).astype("int32")
                # xv, yv = np.meshgrid(x, y)
                if random_baseline:
                    if t.is_tensor(randomstart_sample):
                        randomstart_sample = randomstart_sample.cpu().numpy()
                    randomstart_sample[range(num_samples), random_coords_rows, random_coords_cols] = randomfinish_sample.cpu().numpy()[
                                                                                    range(num_samples),
                                                                                    random_coords_rows,
                                                                                    random_coords_cols]

        importance_meanscores = np.mean(importance_scores, axis=-1)
        unimportance_meanscores = np.mean(unimportance_scores, axis=-1)
        if random_baseline:
            random_meanscores = np.mean(random_scores, axis=-1)

        importance_auc = auc(importance_meanscores)
        unimportance_auc = auc(unimportance_meanscores)
        if random_baseline:
            random_auc = auc(random_meanscores)
        print("[Metric {}] AUC: {}\n".format("importance delete", importance_auc))
        print("[Metric {}] AUC: {}\n".format("Unimportance delete", unimportance_auc))
        if random_baseline:
            print("[Metric {}] AUC: {}\n".format("Random baseline delete", random_auc))

        ## Plot
        if verbose == 1:
            plt.figure(figsize=(12, 7))
            if random_baseline:
                plt.subplot(131)
            else:
                plt.subplot(121)
            plt.plot(np.arange(num_steps + 1) / num_steps, importance_meanscores)
            plt.xlim(-0.1, 1.1)
            plt.ylim(0, 1.05)
            plt.fill_between(np.arange(num_steps + 1) / num_steps, 0, importance_meanscores,
                             alpha=0.4)
            plt.title("Importance Deletion")
            plt.xlabel(f"Percent of deletion \n until {percent * 100} %")
            plt.ylabel("Prediction of model")

            if random_baseline:
                plt.subplot(132)
            else:
                plt.subplot(122)
            plt.plot(np.arange(num_steps + 1) / num_steps, unimportance_meanscores)
            plt.xlim(-0.1, 1.1)
            plt.ylim(0, 1.05)
            plt.fill_between(np.arange(num_steps + 1) / num_steps, 0, unimportance_meanscores,
                             alpha=0.4)
            plt.title("Unimportance Deletion")
            plt.xlabel(f"Percent of deletion \n until {percent * 100} %")
            plt.ylabel("Prediction of model")

            if random_baseline:
                plt.subplot(133)
                plt.plot(np.arange(num_steps + 1) / num_steps, random_meanscores)
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(num_steps + 1) / num_steps, 0, random_meanscores,
                                 alpha=0.4)
                plt.title("Random baseline Deletion")
                plt.xlabel(f"Percent of deletion \n until {percent * 100} %")
                plt.ylabel("Prediction of model")
#             plt.suptitle(f"{method} Temporal Importance Eval. with {self.eval_mode} and percent:{percent} with Saliency {typeofsaliency}")
            plt.suptitle(f"{method} Temporal Importance Eval. with {self.eval_mode} and percent:{percent}")
            if save_to is not None:
                plt.tight_layout()
#                 plt.savefig(save_to + f'/{method}_temporal_importance_eval_{self.eval_mode}_len_{percent}_with_sali_{typeofsaliency}.png')

                plt.savefig(save_to + f'_{method}_temporal_importance_eval_{self.eval_mode}_len_{percent}.png')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()

        importance_auc_persample = np.zeros((num_samples, 1))
        unimportance_auc_persample = np.zeros((num_samples, 1))
        if random_baseline:
            random_auc_persample = np.zeros((num_samples, 1))
        for i in range(num_samples):
            importance_auc_persample[i] = sk_auc(range(num_steps+1), importance_scores[:, i])
            unimportance_auc_persample[i] = sk_auc(range(num_steps+1), unimportance_scores[:, i])
            if random_baseline:
                random_auc_persample[i] = sk_auc(range(num_steps+1), random_scores[:, i])
        if random_baseline:
            auc_persample_list = [importance_auc_persample, unimportance_auc_persample, random_auc_persample]
            scores_list = [importance_scores, unimportance_scores, random_scores]
        else:
            auc_persample_list = [importance_auc_persample, unimportance_auc_persample]
            scores_list = [importance_scores, unimportance_scores]
        ## return dicts
        auc_persample_dict = {}
        scores_dict = {}
        if random_baseline:
            criterions = ['importance', 'unimportance', 'random']
        else:
            criterions = ['importance', 'unimportance']
        for i, criterion in enumerate(criterions):
            auc_persample_dict[criterion] = auc_persample_list[i]
            scores_dict[criterion] = scores_list[i]

        return auc_persample_dict, scores_dict
        # return importance_meanscores, unimportance_meanscores, random_meanscores, importance_auc, unimportance_auc, random_auc


def auc(scores):
    """Calculate the (normalized) Area under Curve"""
    area = np.sum(scores) - scores[0]/2 - scores[-1]/2
    area /= (scores.shape[0] -1)
    return area

def quantile_values_like(quantile: float, dataset):
    """Return the quantile value over the whole dataset
        Has the same shape of a single sample in dataset

        Parameters
        ----------
        quantile (float) : how much percent the value should be
        dataset: the whole testset [Batch, Features(dim), Length]
        Returns
        -------
        quantile_like_array for single sample [1(B), Features(dim), Length]
    """
    ## Each feature should have one quantile value
    quantile_values = np.quantile(dataset, q=quantile, axis=-1)
    quantile_like = np.ones(dataset.shape)
    quantile_like *= np.expand_dims(quantile_values, axis=-1)

    return t.from_numpy(quantile_like.astype("float32"))
