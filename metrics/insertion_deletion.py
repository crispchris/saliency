"""
Metric: Insertion/Deletion
Refer to: https://arxiv.org/abs/1806.07421
References: https://github.com/eclique/RISE/blob/master/Evaluation.ipynb

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




class MetricInsertDelete:
    def __init__(self, model: nn.Module, mode: str, step: int, substrate_fn,
                 device=None):
        """
        Create insertion/deletion metric instance

        Parameters
        ----------
        model (nn.Module): Deep leanring model to be explained
        mode (str) : "ins" or "del"
        step (int) : number of pixels modified per one iteration
        substrate_fn (func) : a mapping from old pixes to new pixels
        device: torch CPU or GPU
        """

        assert mode in ["ins", "del"]
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.device = device

    def single_run(self, sample: np.ndarray, saliency_map: np.ndarray,
                   verbose=0, save_to=None):
        """
        Run metric on a single sample

        Parameters
        ----------
        sample (np.ndarray) : normalized sample [D, L]
        saliency_map (np.ndarray) : saliency map from a vis.Method [D, L]
        verbose (int) : in [0, 1].
            0 - return list of scores (no plot)
            1 - print the top 2 classes
            2 - plot every step
        save_to (str) : directory to save plots

        Returns
        -------
        scores (np.ndarray) : Array containing scores at every step
        """
        sample = t.tensor(sample).float().to(self.device)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        ## prediction of original sample
        pred = self.model(sample) ## pred shape [B, C, L] C: Classes or Features
        prob, idx = t.max(pred, dim=1)
        idx = idx.cpu().numpy()[0]

        if verbose == 2:
            print("The predicted Class {} : Probability {:.4f}".format(idx[0], float(prob[0][0])))

        assert sample.shape[-1] % self.step == 0
        ## number of steps
        num_steps = sample.shape[-1] // self.step

        ## start and finish arrays
        if self.mode == "del":
            print("[Metric] Deletion Mode")
            title = "Deletion Game"
            ylabel = "Features in Time Domain Deleted"
            start_sample = sample.clone().detach()
            finish_sample = self.substrate_fn(sample)
        elif self.mode == "ins":
            print("[Metric] Insertion Mode")
            title = "Insertion Game"
            ylabel = "Features in Time Domain Inserted"
            start_sample = self.substrate_fn(sample)
            finish_sample = sample.clone().detach()
        assert start_sample is not None
        assert finish_sample is not None

        scores = np.empty(num_steps+1)
        ## Coordinates of features(pixels) in order of decreasing saliency
        saliency_map = np.expand_dims(saliency_map, axis=0)
        saliency_order = np.flip(np.argsort(saliency_map, axis=-1), axis=-1)
        ## For testing
        # np.random.seed(42)
        # np.random.shuffle(saliency_order)
        # saliency_order = np.argsort(saliency_map, axis=-1) ## start from the lowest saliency
        # For plot
        ## Coloring dimensions
        color_mapping = plt.get_cmap().colors
        cmap = color_mapping[0::256//sample.shape[1]]

        for i in range(num_steps+1):
            # pred = self.model(start_sample.clone().detach().to(self.device))
            pred = self.model(t.tensor(start_sample).to(self.device))
            prob1, idx1 = t.topk(pred, k=2, dim=1)

            if verbose == 2:
                print("(high) First Class {} : Probability {:.4f}".format(idx1[0][0][0], float(prob1[0][0][0])))
                print("Second Class {} : Probability {:.4f}".format(idx1[0][1][0], float(prob1[0][1][0])))

            scores[i] = pred[0, idx, 0]
            if verbose == 2 or (i == num_steps and verbose == 1):
                if verbose == 2 and i == 0:
                    plt.figure(figsize=(10, 5))
                elif verbose == 1:
                    plt.figure(figsize=(10, 5))

                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / num_steps, scores[i]))
                if t.is_tensor(start_sample):
                    start_sample = start_sample.cpu().numpy()
                for dim in range(start_sample.shape[1]):
                    plt.plot(range(start_sample.shape[-1]), start_sample[0, dim, :],
                             c=cmap[dim])

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / num_steps, scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / num_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel("Class {}".format(idx[0]))
                if save_to is not None:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()

            if i < num_steps:
                coords = saliency_order[:, :, i*self.step:(i+1)*self.step]

                x = np.linspace(0, 1, coords.shape[2]).astype("int32")
                y = np.linspace(0, coords.shape[1] - 1, coords.shape[1]).astype("int32")
                xv, yv = np.meshgrid(x, y)
                if t.is_tensor(start_sample):
                    start_sample = start_sample.cpu().numpy()
                start_sample[:, yv, coords] = finish_sample.cpu().numpy()[:, yv, coords]

        if save_to is not None:
            plt.savefig(save_to + '/insertion_deletion_game.png')
            plt.close()
        else:
            plt.show()
        return scores



    def evaluation(self, batch_samples: np.ndarray,
                   batch_labels: List,
                   batch_saliency_maps: np.ndarray,
                   batch_size: int):
        """
        Evaluation an explaination on a dataset and return the average scores

        Parameters
        ----------
        batch_samples (np.ndarray): batch of samples [Batch, Features(dim), Length]
        batch_labels (List): batch of Labels (corresponding to batch_samples)
        batch_saliency_maps (np.ndarray): batch of explanations [Batch, Features(dim), Length]
        batch_size (int): number of sample for one batch

        Returns
        -------
        scores (np.ndarray): Array containing scores at every step [Batch, num of steps]
        """
        num_samples = batch_samples.shape[0]
        num_classes = len(np.unique(batch_labels))
        assert num_samples % batch_size == 0
        predictions = t.FloatTensor(num_samples, num_classes)

        batch_samples = t.tensor(batch_samples).float().to(self.device)

        for i in tqdm(range(num_samples // batch_size), desc= "Predicting Labels"):
            preds = self.model(batch_samples[i*batch_size:(i+1)*batch_size])
            predictions[i*batch_size:(i+1)*batch_size] = preds.squeeze(dim=-1)
        top = np.argmax(predictions.detach().numpy(), axis=1)

        assert batch_samples.shape[-1] % self.step == 0
        num_steps = batch_samples.shape[-1] // self.step
        scores = np.empty((num_steps+1, num_samples))

        saliency_orders = np.flip(np.argsort(batch_saliency_maps, axis=-1), axis=-1)
        ## For testing
        # np.random.seed(42)
        # np.random.shuffle(saliency_orders)
        # saliency_orders = np.argsort(batch_saliency_maps, axis=-1)  ## start from the lowest saliency
        ## start and finish arrays
        if self.mode == "del":
            print("[Metric] Deletion Mode")
            caption = "delete"
            title = "Deletion Game"
            ylabel = "Features in Time Domain Deleted"
            start_bsamples = batch_samples.clone().detach()
            finish_bsamples = self.substrate_fn(batch_samples)
        elif self.mode == "ins":
            print("[Metric] Insertion Mode")
            caption = "insert"
            title = "Insertion Game"
            ylabel = "Features in Time Domain Inserted"
            start_bsamples = self.substrate_fn(batch_samples)
            finish_bsamples = batch_samples.clone().detach()
        assert start_bsamples is not None
        assert finish_bsamples is not None

        for i in tqdm(range(num_steps+1), desc= caption + "features in Length"):
            # Iterate over batches
            for j in range(num_samples // batch_size):
                # Compute prediction for new samples (adjusted)
                preds = self.model(t.tensor(start_bsamples[j*batch_size:(j+1)*batch_size]).to(self.device))
                preds = preds.detach().cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds.reshape(-1)
            ## find the coordination for the saliency maps
            coords = saliency_orders[:, :, i*self.step:(i+1)*self.step]

            x = np.linspace(0, 1, coords.shape[2]).astype("int32")
            y = np.linspace(0, coords.shape[1] - 1, coords.shape[1]).astype("int32")
            xv, yv = np.meshgrid(x, y)
            if isinstance(start_bsamples, t.Tensor):
                start_bsamples = start_bsamples.cpu().numpy()
            start_bsamples[:, yv, coords] = finish_bsamples.cpu().numpy()[:, yv, coords]

        mean_scores = np.mean(scores, axis=-1)
        area_under_curve = auc(mean_scores)
        print("[Metric {}] AUC: {}\n".format(title, area_under_curve))
        ## plot
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(num_steps+1) / num_steps, mean_scores)
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)
        plt.fill_between(np.arange(num_steps+1) / num_steps, 0, mean_scores, alpha=0.4)
        plt.title(title)
        plt.xlabel(ylabel)
        plt.ylabel("Whole Data Evaluation")
        plt.show()

        auc_persample = np.zeros((num_samples, 1))
        for i in range(num_samples):
            auc_persample[i] = sk_auc(range(num_steps+1), scores[:, i])



        return mean_scores, area_under_curve, auc_persample

def auc(scores):
    """Calculate the (normalized) Area under Curve"""
    area = np.sum(scores) - scores[0]/2 - scores[-1]/2
    area /= (scores.shape[0] -1)
    return area

### substrate function
def gaussian_kernel(klen, nsig):
    """
    Returns a Gaussian Kernel
    Convolution with it results in signal Blurring
    Parameters
    ----------
    klen: kernel length
    nsig: noise sigma (standard deviation)

    Returns
    -------
    Gaussian kernel (Tensor)
    """
    inp = np.zeros((1, klen))
    ## set element at the middle to one, a dirac delta
    inp[0, klen//2] = 1
    kernel = gaussian_filter1d(inp, sigma=nsig)
    return t.from_numpy(kernel.astype("float32"))

def gaussian_kernel_(klen, nsig, datasample:np.ndarray):
    """
    Returns a Gaussian Kernel
    Convolution with it results in signal Blurring(Smooth)
    Parameters
    ----------
    klen: kernel length
    nsig: noise sigma (standard deviation)
    datasample (np.ndarray): sample of data [C, L] (to get to know the shape of channels)

    Returns
    -------
    Gaussian kernel (Tensor)
    """
    feature_dim = datasample.shape[0]
    gkernel = t.zeros((feature_dim, feature_dim, klen))
    inp = np.zeros((1, klen))
    ## set element at the middle to one, a dirac delta
    inp[0, klen//2] = 1
    kernel = t.from_numpy(gaussian_filter1d(inp, sigma=nsig).astype("float32"))
    for i in range(feature_dim):
        gkernel[i, i, :] = kernel.squeeze(dim=0)
    return gkernel


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
#
# def random_values_like(dataset):
#     """Return a Zero tensor with the size of dataset"""
#     min_ = np.min(dataset, axis=-1)
#     max_ = np.max(dataset, axis=-1)
#     for i in range(dataset.shape[-1]):
#         random_like =
#     return t.from_numpy(zero_like.astype("float32"))

