import numpy as np 


def entropy(pX):
    """
    This function calculates the entropy given an input dataset

    Args:
        pX ([numpy.ndarray]): Input probability mass distribution

    Returns:
        numpy.float64: It is the computed entropy 
    """

    'I find the unique values of x in order to get the mass function of x'
    
    return np.sum(-pX * np.log2(pX)) # Return the entropy computed as defined by Shannon 


def entropy_joint(pXY):
    """[summary]

    Args:
        x ([type]): [description]
        numBins (int, optional): [description]. Defaults to 30.

    Returns:
        [type]: [description]
    """

    return -np.sum(pXY * np.log2(pXY))


def conditional_entropy(pXY, pX):
    """[summary]

    Args:
        pXY ([type]): [description]
        pX ([type]): [description]

    Returns:
        [type]: [description]
    """
    pY_givenX = pXY / pX
    return - np.sum(pXY * np.log2(pY_givenX)) 


def mutual_information(pXY, pX, pY):
    """[summary]

    Args:
        pXY ([type]): [description]
        pX ([type]): [description]
        pY ([type]): [description]
    """
    
    return entropy(pX) + entropy(pY) - entropy_joint(pXY)