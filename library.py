import numpy as np 


def entropy(x):
    """
    This function calculates the entropy given an input dataset

    Args:
        x ([numpy.ndarray]): Input data

    Returns:
        numpy.float64: It is the computed entropy 
    """

    'I find the unique values of x in order to get the mass function of x'
    value, counts = np.unique(x, return_counts=True) 

    del value # I don't need this information to compute the entropy
    p = counts / counts.sum() # calculate the probability mass function
    return np.sum(-p * np.log2(p)) # Return the entropy computed as defined by Shannon 


def entropy_joint(x, numBins=30):
    """[summary]

    Args:
        x ([type]): [description]
        numBins (int, optional): [description]. Defaults to 30.

    Returns:
        [type]: [description]
    """

    p, _ = np.histogramdd(x, bins=30)
    p /= p.sum()
    return np.sum(-p * np.log2(p))