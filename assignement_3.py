import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from library.pmf_estimators import *
from library.information_discrete import MI, E


df = sns.load_dataset("iris")
df_discr = df.copy()

# Discretizing the Dataset
df_discr.iloc[:, :-1] = (df_discr.iloc[:, :-1] * 10).astype(int)


# defining some constants to use in our purposes
features = df.columns.to_list()[:-1]
rows, n_features = df_discr.iloc[:, :-1].shape

# Computing the Entropies for all the Iris features
Entropies = []
for feature in features:
    x_vals, pmf = pmf_univariate(df_discr[feature])  # Computing the pmf
    Entropies.append(E(pmf))

for i in range(n_features):
    print(f'{features[i]}: {Entropies[i]}\n')


MI_matrix = np.zeros((n_features, n_features))

# Computing the Mutual information over all the features combinations
for i in combinations(range(n_features), 2):
    ix, iy = i                          # extracting the indices 
    Lx, Ly = features[ix], features[iy] # extracting the features
    PXY = Joint_p(df_discr[Lx], df_discr[Ly]) # Computing the Joint Probability
    PX = PXY.sum(axis=1)  # Computing the X marginal probability
    PY = PXY.sum(axis=0)  # Computing the Y marginal probability
    MI_matrix[ix, iy] = MI_matrix[iy, ix] = MI(PXY, PX, PY)  # Mutual Information is symmetric


# Converting the result into a pandas dataframe
MI_matrix = pd.DataFrame(MI_matrix, columns=features, index=features)
MI_matrix.columns.name = 'Mutual Information'
MI_matrix[MI_matrix == 0] = Entropies # adding the entropies to the result
print(MI_matrix)