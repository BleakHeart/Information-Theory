import numpy as np
from library import entropy
import matplotlib.pyplot as plt

# Defining the Probabilities and the Entropy list
P = np.linspace(0, 1, 100)  # p0 array probabilities
S = []                      # Entropy empty list


# Computing the entropy for the p0 values contained in P array
for p0 in P:
    # Creating a binary random array with p0 and 1-p0 probabilities of 280000 shape.
    data = np.random.choice([0, 1], p=[p0, 1 - p0], size=280000) 

    # computing the data probabilities
    values, counts = np.unique(data, return_counts=True)
    pdf = counts / counts.sum()

    # saving the entropy value for this set of data
    S.append(entropy(pdf))


# Plotting the results
plt.plot(P, S)
plt.xlabel(r'$p_0$')
plt.ylabel('Entropy (S)')
plt.show()