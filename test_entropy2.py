import numpy as np
from library_discrete import E
import matplotlib.pyplot as plt
from library_continuos import diff_E
from scipy.stats import norm
import seaborn as sns


P = np.linspace(0, 1, 100)  # p0 array probabilities
S_estimated = []                      # Entropy empty list

P_real = np.vstack([P, 1 - P]).T

# Computing the entropy for the p0 values contained in P array
for p in P_real:
    # Creating a binary random array with p0 and 1-p0 probabilities of 280000 shape.
    data = np.random.choice([0, 1], p=p, size=20000) 

    # computing the data probabilities
    values, counts = np.unique(data, return_counts=True)
    pdf = counts / counts.sum()

    # saving the entropy value for this set of data
    S_estimated.append(E(pdf))

# Computing the real Entropy to see the differences
S_real = [E(p) for p in P_real]


# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(12, 9))

axs[0].plot(P, S_real, 'k')
axs[0].set_xlabel(r'$p_0$')
axs[0].set_ylabel('Entropy (S)')
axs[0].set_title('Results from the real pmf')

axs[1].plot(P, S_estimated, 'k')
axs[1].set_xlabel(r'$p_0$')
axs[1].set_ylabel('Entropy (S)')
axs[1].set_title('Results from the estimated pmf')

plt.show()