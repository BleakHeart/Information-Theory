import numpy as np
from library.information_discrete import E
import matplotlib.pyplot as plt
from library.information_continuos import diff_E
from scipy.stats import norm
import seaborn as sns


x = np.linspace(-5, 5, 100)
dxReal = x[1] - x[0]
pdfReal = norm.pdf(x)

n_generated = 100000  # samples number to generate
E_pdf = []


# Computing the difference between the true pdf and the estimated one
for i in range(1000):
    # generating the sample
    samples = np.random.normal(size=n_generated)

    # calculating optimal number of bins with the Scott's Rule
    n_bins = int(3.49 * np.std(samples) * n_generated ** (1/3))

    pdfEstimated, bins = np.histogram(samples, bins=n_bins, density=True)
    dxEstimated = bins[1] - bins[0]

    E_pdf.append(diff_E(pdfEstimated, dxEstimated))


# increasing the samples size, the difference tends to zero
sns.boxplot(x=E_pdf)
plt.xlabel('Estimated Entropy')
plt.title(f'Real pdf Entropy: {diff_E(pdfReal, dxReal):.4f}')
plt.show()