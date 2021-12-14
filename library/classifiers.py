import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.nonparametric.kernel_density import KDEMultivariate, KDEMultivariateConditional
import pandas as pd


class Bayes_Classifier:
    def __init__(self, X, y, dep_type, indep_type, bw=None) -> None:
        self.n_classes = np.unique(y).size
        self.n_features = X.shape[1]

        self.Prior = np.unique(y, return_counts=True)[1] / y.size

        if bw == None:
            bw = 'normal_reference'
        else:
            bw = bw
        
        self.kde_Xc = KDEMultivariateConditional(endog=X, exog=y,
                                        dep_type=dep_type, indep_type=indep_type,
                                        bw=bw).pdf
        self.kde_X = KDEMultivariate(data=X, var_type=dep_type).pdf

    def evaluate(self, X_test):
        n = X_test.shape[0]

        PXc = np.array([self.kde_Xc(X_test, [i] * n) for i in range(3)])
        PcX = ((PXc.T * self.Prior).T / self.kde_X(X_test)).T

        return PcX


def Naive_Bayes_classifier(X, y, dep_type, indep_type):
    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    kde_Xc = []

    for i in range(n_features):
        kde_Xc.append(KDEMultivariateConditional(endog=X[:, i], exog=y, 
                                                 dep_type=dep_type, 
                                                 indep_type=indep_type, 
                                                 bw='normal_reference').pdf)
        
    kde_X = KDEMultivariate(X, var_type='cccc').pdf

    Prior = np.unique(y, return_counts=True)[1] / y.size

    c = np.zeros(n_classes)
    for i in range(n_classes):
        tmp = 1
        for j in range(n_features):
            tmp *= kde_Xc[j]([X[0, j]], [i])
        c[i] = tmp * Prior[i] / kde_X(X[0, :])

    return c