"""
Localization: with Human perspective, bounding box from labeling (from image processing)
            Can in time series Data (Apply on densely labeling)
(Metrics)
Pointing Game:
PG = #Hits / (#Hits + #Misses)

the Ratio of Intersection between the salient area and the ground truth mask over the Salient Region
(IoSR)
IoSR = (Mgt includes with SA(E(I,f)) / SA(E(I,f))
SA(E(I,f) is  as SUM(SA(E(I,f) > theta * max(SA(E(I,f)))
theta is a user-defined threshold

"""
import operator
import numpy as np


def pointing_game(saliency_map: np.ndarray,
                  targets,
                  labels
                  ):
    """
    Suitable for densely labeling in Time series data

    Parameters
    ----------
    saliency_map (np.ndarray): the explanation, shape = [#Batch, #Feature, #Length]
    targets: the real target, which produced the saliency maps
    labels: the real densely labels from the data and saliency maps

    Returns
    -------
    pointing game
    """
    targets = targets.cpu().detach().numpy()
    if saliency_map.shape[0] != len(targets):
        raise ValueError("the batch size of saliency map should be the same as the length of targets")
    if saliency_map.shape[0] != len(labels):
        raise ValueError("the batch size of saliency map should be the same as the length of labels")

    num_hits = 0
    num_misses = 0
    num_test_sample = 0
    for i in range(saliency_map.shape[0]):
        mask = labels[i] == targets[i]
        mask_sum = np.sum(mask)
        if mask_sum != 0:
            num_test_sample += 1

            max_idx = np.argmax(saliency_map[i, :, :], axis=-1) ## index of the max ## length of features as output
            mask_idx = np.where(mask == True)[0]
            # mask_idx = [x for x in range(len(mask)) if mask[x] == True]
            for each_max in max_idx:
                if each_max in mask_idx:
                    num_hits += 1
                else:
                    num_misses += 1

    print("[Number of Test samples] Number: ", num_test_sample)
    pointinggame_score = num_hits / (num_hits + num_misses)
    return pointinggame_score, num_test_sample

def iosaliency_regions(saliency_map: np.ndarray,
                       targets,
                       labels,
                       threhold: float = 0.5
                       ):
    """
        Suitable for densely labeling in Time series data

        Parameters
        ----------
        saliency_map (np.ndarray): the explanation, shape = [#Batch, #Feature, #Length]
        targets: the real target, which produced the saliency maps
        labels: the real densely labels from the data and saliency maps
        threhold (float): default = 0.5, to decide which saliency pixel is over the threshold
                            (threshold * max_saliency_value)
        Returns
        -------
        pointing game
    """
    targets = targets.cpu().detach().numpy()
    if saliency_map.shape[0] != len(targets):
        raise ValueError("the batch size of saliency map should be the same as the length of targets")
    if saliency_map.shape[0] != len(labels):
        raise ValueError("the batch size of saliency map should be the same as the length of labels")

    iosrs_list = []
    num_sample_tested = 0
    for i in range(saliency_map.shape[0]):
        num_hits = 0
        mask = labels[i] == targets[i]
        mask_sum = np.sum(mask)
        if mask_sum != 0:
            max_Explain = np.max(saliency_map[i, :, :]) ## the max of the explanation
            # max_idx = np.argmax(saliency_map[i, :, :], axis=-1) ## index of the max ## length of features as output
            # saliency_map_list = saliency_map[i].tolist()
            # max_values = [operator.itemgetter(int(idx))(sm) for sm, idx in zip(saliency_map_list, max_idx)]
            threhold_values = max_Explain * threhold
            threhold_values_2d = [[threhold_values] * saliency_map[i, j, :].shape[-1]
                                  for j in range(saliency_map[i].shape[0])]
            # threhold_values_2d = [[threhold_values[i]] * saliency_map[i, :, :].shape[-1] for i in range(len(threhold_values))]

            mask_all_over_thres = saliency_map[i, :, :] > threhold_values_2d
            num_all_over_thres = np.sum(mask_all_over_thres) ## denominator
            idx_all_over_thres = np.where(mask_all_over_thres == True)

            mask_idx = np.where(mask == True)[0]
            # mask_idx = [x for x in range(len(mask)) if mask[x] == True]
            # for fea in range(idx_all_over_thres.shape[0]):
            if num_all_over_thres != 0:
                num_sample_tested += 1
                for idx in idx_all_over_thres[1]: ## cols idxs
                    if idx in mask_idx:
                        num_hits += 1
                iosr = num_hits / num_all_over_thres
                iosrs_list.append(iosr)
    print("[Number of Test samples] Number: ", num_sample_tested)
    mean_iosr = np.mean(iosrs_list)
    std_iosr = np.std(iosrs_list)
    return mean_iosr, std_iosr, num_sample_tested
