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


# Computing the difference between the true pdf and the estimated one

x = np.linspace(-5, 5, 100)
dxReal = x[1] - x[0]
pdfReal = norm.pdf(x)

n_generated = 10000
E_pdf = []

for i in range(1000):
    samples = np.random.normal(size=n_generated)

    # calculating optimal number of bins with the Scott's Rule
    n_bins = int(3.49 * np.std(samples) * n_generated ** (1/3))

    pdfEstimated, bins = np.histogram(samples, bins=n_bins, density=True)
    dxEstimated = bins[1] - bins[0]

    E_pdf.append(diff_E(pdfEstimated, dxEstimated))


# increasing the samples size, the difference tends to zero

sns.boxplot(x=E_pdf)
plt.xlabel('Estimated Entropy')
plt.title(f'Real pdf Entropy: {diff_E(pdfReal, dxReal):.2f}')
plt.show()