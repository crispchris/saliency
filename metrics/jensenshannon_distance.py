"""
Compute distance between two probability distributions
"""
import numpy as np
import scipy.stats
def jensen_shannon_distance(p, q, base = 2, axis=1):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m, base=base, axis=axis) + scipy.stats.entropy(q, m, base=base, axis=axis)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance