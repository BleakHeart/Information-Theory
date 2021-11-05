import numpy as np
from scipy.special import xlogy

def diff_E(pdf, dx):
    return - (xlogy(pdf, pdf) * dx).sum()