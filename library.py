import numpy as np 


def entropy(pX):
    """
    This function calculates the entropy given an input dataset

    Args:
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Entropy computed 
    """

    'I find the unique values of x in order to get the mass function of x'
    
    return np.nansum(-pX * np.log2(pX)) # Return the entropy computed as defined by Shannon 


def entropy_joint(pXY):
    """
    This Function computes the joint entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability

    Returns:
        numpy.float64: Joint Entropy computed
    """

    return -np.nansum(pXY * np.log2(pXY))


def conditional_entropy(pXY, pX):
    """
    This Function computes the conditional entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Conditional Entropy
    """
    pY_givenX = pXY / pX
    return - np.nansum(pXY * np.log2(pY_givenX))


def mutual_information(pXY, pX, pY):
    """
    This Function computes the Mutual Information using 
    one of the relationships with the Entropy.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        pX ([numpy.array]): Probability mass function of X
        pY ([numpy.array]): Probability mass function of Y

    Returns:
        numpy.float64: Mutual Information
    """
    
    return entropy(pX) + entropy(pY) - entropy_joint(pXY)