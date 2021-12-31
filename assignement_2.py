import numpy as np
import matplotlib.pyplot as plt
from library.information_continuos import diff_E
from library.pdf_estimators import *
from scipy.stats import norm
import seaborn as sns
from library.plot import plot_settings, plot_kernels


xReal = np.linspace(-5, 5, 10000)
dxReal = xReal[1] - xReal[0]
pdfReal = norm.pdf(xReal)

RealEntropy = diff_E(pdfReal, xReal)

n_generated = 10000  # samples number to generate
E_pdf = []
kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

n_samples = [np.power(10, i) for i in range(1, 6)]
results = np.zeros((len(n_samples), len(kernels)))
for i, kernel in enumerate(kernels):
    E_pdf = []
    for j, n in enumerate(n_samples):
        samples = np.random.normal(size=n)
        x, pdfEstimated = kde_sklearn(samples, kernel)

        E_pdf.append(diff_E(pdfEstimated, x))

    print(f'fatto kernel {kernel}')
    results[:, i] = E_pdf

df = pd.DataFrame({'N generated': n_samples})

df[kernels] = results
df = df.set_index('N generated')
df

plot_settings()
fig, axs = plt.subplots(1, 2)
plot_kernels(axs[0])

df.plot(logx=True, ax=axs[1], lw=0.5)
axs[1].plot(n_samples, [diff_E(pdfReal, xReal)]* len(n_samples), label='exact pdf', lw=0.5)
axs[1].legend()
axs[1].grid()
axs[1].set_title('Differential Entropy')
fig.tight_layout()
#plt.savefig('./Images/Kernels_diffential', dpi=600)

print(f'L\'entropia teoria vale: {RealEntropy}')