import numpy as np 
import pandas as pd
from sklearn.neighbors import KernelDensity


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

    mixture = kernel(u).sum(axis=0)
    mixture /= np.abs(mixture).max()

    return mixture

def kde_univariate(data, kernel):
    """function to estimate the probability density function
       given a feature and a kernel.

    Args:
        data (np.array): Data used to estimate the density function
        kernel (str): kernel used to estimate the density function
    """
    if isinstance(data, pd.Series):
        data = data.to_numpy()


    iqr = np.subtract(*np.percentile(data, [75, 25]))
    m = np.min([np.sqrt(np.var(data)), iqr / 1.349])
    h = 0.9 * m / np.power(data.size, 1/5)  # Silvermann's optimum estimate

    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(data)
    x_domain = np.linspace(min(data), max(data), 1000)
    pdf = np.exp(kde.score_samples(x_domain))
    
    return x_domain, pdf