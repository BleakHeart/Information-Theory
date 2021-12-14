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

        PXc = np.array([self.kde_Xc(X_test, [i] * n) for i in range(self.n_classes)])
        PcX = ((PXc.T * self.Prior).T / self.kde_X(X_test)).T

        return PcX


class NB_classifier:
    def __init__(self, X, y, dep_type, indep_type, bw=None) -> None:
        self.n_classes = np.unique(y).size
        self.n_features = X.shape[1]

        self.Prior = np.unique(y, return_counts=True)[1] / y.size

        if bw == None:
            bw = 'normal_reference'
        else:
            bw = bw

        self.kde_Xc = []

        for i in range(self.n_features):
            self.kde_Xc.append(KDEMultivariateConditional(endog=X[:, i], exog=y, 
                                                     dep_type=dep_type, 
                                                     indep_type=indep_type, 
                                                     bw='normal_reference').pdf)
        
        self.kde_X = KDEMultivariate(X, var_type='cccc').pdf

        self.Prior = np.unique(y, return_counts=True)[1] / y.size
    
    def evaluate(self, X):
        n_rows = X.shape[0]
        c = np.zeros((n_rows, self.n_classes))
        
        for d in range(n_rows):
            for i in range(self.n_classes):
                tmp = 1
                for j in range(self.n_features):
                    tmp *= self.kde_Xc[j]([X[d, j]], [i])
                
                c[d, i] = tmp * self.Prior[i] / self.kde_X(X[d, :])
        return c