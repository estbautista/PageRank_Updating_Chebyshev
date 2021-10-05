import numpy as np
import scipy.stats as spt
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

data_pr = np.load('Res_Fig5_stdPR.npz')

error_update = data_pr['error_update']
error_scratch = data_pr['error_scratch']
perturbation_size = data_pr['perturbation_size']

fig = plt.figure(figsize=[7.5,3.5])
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

### Top Figure
ax0 = plt.subplot(gs[0])
ax0.plot(np.arange(1, len(error_update)), error_update[1:], label='Update', linestyle='solid', marker='None', color='tab:red', markersize=6, linewidth=3, markerfacecolor='None', markeredgewidth=2)
ax0.plot(np.arange(len(error_update)), error_scratch,  label='Scratch', linestyle='solid', marker='None', color='tab:blue', markersize=6, linewidth=3, markerfacecolor='None', markeredgewidth=2)
ax0.legend(fontsize=11)
ax0.yaxis.set_tick_params(labelsize=11)
plt.grid('True', linewidth=0.5)
plt.yscale('log')
plt.ylabel('Error', fontsize=13)
plt.ylim([1e-13,1e-7])

### Bottom figure
ax1 = plt.subplot(gs[1])
ax1.plot(np.arange(1, len(error_update)+1), -perturbation_size, linestyle='solid', marker='None', color='black', linewidth=0.5)
plt.setp(ax0.get_xticklabels(), visible=False)
yticks = ax1.yaxis.get_major_ticks()
plt.yticks([0,-20,-40])
ax1.yaxis.set_tick_params(labelsize=10)
plt.subplots_adjust(hspace=.05)
plt.ylabel('Leaving \nEdges', fontsize=13)
plt.xlabel('Snapshot', fontsize=13)
plt.rcParams.update({'font.size': 11})
plt.savefig('fig5_stdpr', dpi=300, edgecolor='w',bbox_inches="tight")
