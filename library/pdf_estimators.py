import numpy as np 
import pandas as pd

def gaussian(u):
    """Gaussian kernel

    Args:
        u (np.array): Set values
    """
    
    return np.exp(-0.5 * np.power(u, 2)) / np.power(2 * np.pi, 0.5)


def kde(data, kernel):
    """Function to calculate the Kernel Density Estimator of a sample data

    Args:
        data (np.array): sample data
        kernel (Callable): kernel to use to estimate the pdf
    """
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    N = len(data)
    data = np.asanyarray(data)

    points = np.linspace(min(data), max(data), 1000).reshape((1, -1))
    points = np.concatenate([points] * N)

    iqr = np.subtract(*np.percentile(data, [75, 25]))
    m = np.min([np.sqrt(np.var(data)), iqr / 1.349])
    h = 0.9 * m / np.power(data.size, 1/5)  # Silvermann's optimum estimate

    u = (points - data[:, np.newaxis]) / h

    mixture = gaussian(u).sum(axis=0)
    mixture /= np.abs(mixture).max()

    return mixture
