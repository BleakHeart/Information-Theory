import numpy as np
from library_discrete import E
import matplotlib.pyplot as plt
from library_continuos import diff_E
from scipy.stats import norm


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
    S.append(E(pdf))


# Plotting the results
plt.plot(P, S)
plt.xlabel(r'$p_0$')
plt.ylabel('Entropy (S)')
plt.show()


# Computing the difference between real pmf and the estimated one 

pmfReal = [0.30, 0.70]

samples = np.random.choice([0, 1], p=pmfReal, size=100)
_, pmfEstimated = np.unique(samples, return_counts=1)
pmfEstimated = pmfEstimated / pmfEstimated.sum()

print(E(pmfReal) - E(pmfEstimated))


# Computing the difference between the true pdf and the estimated one

x = np.linspace(-5, 5, 100)
dxReal = x[1] - x[0]
pdfReal = norm.pdf(x)

samples = np.random.normal(size=1000)
pdfEstimated, bins = np.histogram(samples, bins=300, density=True)
dxEstimated = bins[1] - bins[0]

# increasing the samples size, the difference tends to zero
print(diff_E(pdfReal, dxReal) - diff_E(pdfEstimated, dxEstimated))