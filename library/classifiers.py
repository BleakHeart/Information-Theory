from _typeshed import Self
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.nonparametric.kernel_density import KDEMultivariate, KDEMultivariateConditional
import pandas as pd


class Bayes_Classifier:
    def __init__(self, X, y, dep_type, indep_type, bw=None) -> None:
        self.X = X
        self.y = y
        self.dep_type = dep_type
        self.indep_type = indep_type

        self.n_classes = np.unique(y).size
        self.n_features = X.shape[1]

        self.Prior = np.unique(y, return_counts=True)[1] / y.size

        if bw == None:
            self.bw = 'normal_reference'
        else:
            self.bw = bw
        
    def fit(self):
        self.kde_Xc = KDEMultivariateConditional(endog=self.X, exog=self.y,
                                        dep_type=self.dep_type, indep_type=self.indep_type,
                                        bw=self.bw).pdf
        self.kde_X = KDEMultivariate(self.X, 
                                     var_type=self.dep_type).pdf

    def evaluate(self, X_test):
        n = X_test.shape[0]

        PXc = np.array([self.kde_Xc(X_test, [i] * n) for i in range(3)])
        PcX = ((PXc.T * self.Prior).T / self.kde_X(X_test)).T

        return PcX

