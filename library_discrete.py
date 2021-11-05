import numpy as np
from scipy.special import xlogy


def E(pX):
    """
    This function calculates the entropy given an input dataset

    Args:
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Shannon Entropy in nats units
    """

    return -xlogy(pX, pX).sum()


def EJ(pXY):
    """
    This Function computes the joint entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        normalized

    Returns:
        numpy.float64: Joint Entropy in nats
    """
    E_joint = -xlogy(pXY, pXY).sum()

    return E_joint


def CE(pXY, pX):
    """
    This Function computes the conditional entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Conditional Entropy using one of its properties
    """

    return EJ(pXY) - E(pX)


def MI(pXY, pX, pY):
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
    
    return E(pX) + E(pY) - E(pXY)


def EJ_norm(pXY):
    """
    This Function computes the joint entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability

    Returns:
        numpy.float64: Normalized Joint Entropy in nats
    """
    E_joint = EJ(pXY, pXY)

    # Calculating the marginal Probabilities of X and Y
    pX = pXY.sum(axis=0)
    pY = pXY.sum(axis=1)

    return E_joint / (E(pX) + E(pY))


def CE_norm(pXY, pX):
    """
    This Function computes the conditional entropy using its definition.

    Args:
        pXY ([numpy.ndarray]): Joint Probability
        pX ([numpy.array]): Probability mass function of X

    Returns:
        numpy.float64: Normalized Conditional Entropy
    """

    pY_givenX = pXY / pX
    EYgivenX = -xlogy(pXY, pY_givenX).sum()
    return ( EYgivenX )/ E(pX)


def MI_norm(pXY, pX, pY):
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
    
    return ( MI(pXY, pX, pY) ) / EJ(pXY)