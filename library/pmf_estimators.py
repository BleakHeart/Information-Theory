import numpy as np
import pandas as pd

def pmf_univariate(x):
    """function used to compute the pmf of the x array

    Returns:
        np.array: Probability Mass Function of x
    """

    n = x.size # number of x elements
    unique_data, p = np.unique(x, return_counts=True) # determining the unique values and their frequencies
    return unique_data, p/n


def Joint_p(X, Y):
    """
    Compute the joint probability of X and Y

    Args:
        X (np.array) : X feature
        Y (np.array) : Y feature

    Returns:
        np.ndarray: Joint Probability
    """
    return pd.crosstab(X, Y, normalize=True).to_numpy()