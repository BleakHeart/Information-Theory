import numpy as np 
from scipy.special import xlogy


def entropy(pX):
    """
    This function calculates the entropy given an input dataset

    Args:
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Shannon Entropy in nats units
    """

    return -xlogy(pX, pX)


def entropy_joint(pXY, normalized=False):
    """
    This Function computes the joint entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        normalized

    Returns:
        numpy.float64: Joint Entropy in nats
    """
    E_joint = -xlogy(pXY, pXY)

    if normalized:
        pX = pXY.sum(axis=0)
        pY = pXY.sum(axis=1)
        return E_joint / (entropy(pX) + entropy(pY))
    else:
        return E_joint


def conditional_entropy(pXY, pX, normalized=False):
    """
    This Function computes the conditional entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Conditional Entropy
    """

    pY_givenX = pXY / pX
    EYgivenX = -xlogy(pXY, pY_givenX)
    
    if normalized:
        return EYgivenX / entropy(pX)
    else:
        return EYgivenX


def mutual_information(pXY, pX, pY, normalized=False):
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
    MI = entropy(pX) + entropy(pY) - entropy_joint(pXY)
    if normalized:
        return MI / entropy_joint(pXY)
    else:
        return MI