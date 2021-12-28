import numpy as np 
import pandas as pd
from statsmodels.nonparametric.kde import KDEUnivariate
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


def kde_sklearn(x, kernel):
    """Kernel Density Estimation with Scikit-learn"""
    x_grid = np.linspace(x.min(), x.max(), 5000)

    iqr = np.subtract(*np.percentile(x, [75, 25]))
    m = np.min([np.sqrt(np.var(x)), iqr / 1.349])
    h = 0.9 * m / np.power(x.size, 1/5)  # Silvermann's optimum estimate

    kde_skl = KernelDensity(bandwidth=h, kernel=kernel)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return x_grid, np.exp(log_pdf)