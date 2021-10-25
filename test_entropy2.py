import numpy as np
from library import entropy
import matplotlib.pyplot as plt

P = np.linspace(0, 1, 100)
S = []
for p0 in P:
    data = np.random.choice([0, 1], p=[p0, 1 - p0], size=100000)
    values, counts = np.unique(data, return_counts=True)
    pdf = counts / counts.sum()
    S.append(entropy(pdf))

plt.plot(P, S)
plt.xlabel(r'$p_0$')
plt.ylabel('Entropy (S)')
plt.show()