import numpy as np
from scipy.special import xlogy

def diff_E(pdf, dx):
    """Computing the differential entropy using its formula

    Args:
        pdf (np.array): array which contains the pdf
        dx (float): integral differential, proportional to the
                    bin width

    Returns:
        float: Differential Entropy
    """
    
    return - (xlogy(pdf, pdf) * dx).sum()