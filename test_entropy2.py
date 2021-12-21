import numpy as np
from library.information_discrete import E
import matplotlib.pyplot as plt
from library.information_continuos import diff_E
from scipy.stats import norm
import matplotlib as mpl

fig_width_pt = 390.0                # Get this from LaTeX using \the\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 14,
          'legend.fontsize': 14,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'figure.figsize': fig_size,  
          'axes.axisbelow': True}

mpl.rcParams.update(params)



P = np.linspace(0, 1, 100)  # p0 array probabilities


P_real = np.vstack([P, 1 - P]).T

# Computing the real Entropy to see the differences
S_real = [E(p) for p in P_real]

# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(12, 9))

axs[0].plot(P, S_real, 'k')
axs[0].set_xlabel(r'$p_0$')
axs[0].set_ylabel('Entropy (S)')
axs[0].set_title('Results from the real pmf')
axs[0].grid()

# Computing the estimated entropy for different numbers of generated samples 
for s in [100, 1000, 10000, 20000]:
    S_estimated = []                      # Entropy empty list

    # Computing the entropy for the p0 values contained in P array
    for p in P_real:
        # Creating a binary random array with p0 and 1-p0 probabilities of 280000 shape.
        data = np.random.choice([0, 1], p=p, size=s) 

        # computing the data probabilities
        values, counts = np.unique(data, return_counts=True)
        pdf = counts / counts.sum()

        # saving the entropy value for this set of data
        S_estimated.append(E(pdf))
    
    axs[1].plot(P, S_estimated, label=f'samples={s}')


axs[1].set_xlabel(r'$p_0$')
axs[1].set_title('Results from the estimated pmf')
axs[1].grid()
fig.tight_layout()
plt.legend()
plt.savefig('/Images/discrete_Entropy', dpi=600, transparent=False)
#plt.show()