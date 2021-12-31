import numpy as np
from scipy.special import xlogy

def simps(x, y):
    '''Approximate the integral of f(x) from a to b by Simpson's rule.

    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.

    Args:
        x (array): Interval of integration [a,b]
        y (array): array of values corresponding to f(x)

    Returns:
        float: Approximation of the integral of f(x) from a to b using
               Simpson's rule with N subintervals of equal length.
    '''
    N = len(x)
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = x[1] - x[0]
    S = dx/3 * np.sum(y[0:-2:2] + 4*y[1:-1:2] + y[2::2])
    return S


def diff_E(pdf, x):
    """Computing the differential entropy using its formula

    Args:
        pdf (np.array): array which contains the pdf
        dx (float): integral differential, proportional to the
                    bin width

    Returns:
        float: Differential Entropy
    """
    entropies = -xlogy(pdf, pdf)

    return simps(x, entropies)