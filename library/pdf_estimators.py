from typing import Callable
import numpy as np 

def gaussian(u):
    """Gaussian kernel

    Args:
        u (np.array): Set values
    """
    
    return np.exp(-0.5 * np.power(u, 2)) / np.power(2 * np.pi, 0.5)


def kde(data: np.array, kernel: Callable):
    """Function to calculate the Kernel Density Estimator of a sample data

    Args:
        data (np.array): sample data
        kernel (Callable): kernel to use to estimate the pdf
    """

    mixture = np.zeros(1000)
    points = np.linspace(min(data), max(data), 1000)

    iqr = np.subtract(*np.percentile(data, [75, 25]))
    m = np.min([np.sqrt(np.var(data)), iqr / 1.349])
    h = 0.9 * m / np.power(data.size, 1/5)  # Silvermann's optimum estimate

    for xi in data:
        u = (points - xi) / h
        mixture += kernel(u)
    
    mixture /= np.abs(mixture).max()

    return mixture
