import numpy as np
import matplotlib.pyplot as plt
from library.information_continuos import diff_E
from library.pdf_estimators import *
from scipy.stats import norm
import seaborn as sns


x = np.linspace(-5, 5, 10000)
dxReal = x[1] - x[0]
pdfReal = norm.pdf(x)

n_generated = 10000  # samples number to generate
E_pdf = []
kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']


# Computing the difference between the true pdf and the estimated one
n_sim = 100
results = np.zeros((len(kernels), 3))
for j, kernel in enumerate(kernels):
    E_pdf = []
    # generating the sample
    samples = np.random.normal(size=(n_generated, n_sim))
    for i in range(n_sim):
        x, pdfEstimated = kde_sklearn(samples[:, i], kernel)

        dx = x[1] - x[0]

        E_pdf.append(diff_E(pdfEstimated, dx))
    print(f'Fatto Kernel: {kernel}')
    
    results[j, :] = (np.mean(E_pdf), *np.quantile(E_pdf, [0.25, 0.75]))

res = pd.DataFrame({'Kernel': kernels, 'Mean Entropy': results[:, 0], 
                    'q1 Entropy': results[:, 1], 'q2 Entropy': results[:, 2]})

res.to_csv('continuos_entropy_kernels.csv', index=None)