import numpy as np
import scipy.stats as spt
import matplotlib.pyplot as plt

data_pr = np.load('Res_Fig2_stdPR.npz')

messages_update = data_pr['transmissions_update']
messages_scratch = data_pr['transmissions_scratch']
perturbation_size = data_pr['perturbation_size']

plt.figure(figsize=[7.5,2.5])
plt.plot(perturbation_size, messages_update, label='Update', linestyle='-', marker='o', color='tab:red', markersize=7, linewidth=2, markerfacecolor='None', markeredgewidth=2)
plt.plot(perturbation_size, messages_scratch, label='Scratch', linestyle='-', marker='^', color='tab:blue', markersize=7, linewidth=2, markerfacecolor='None', markeredgewidth=2)
plt.legend(loc='lower right')
plt.ylabel('Messages', fontsize=13)
plt.xlabel('New edges', fontsize=13)
plt.grid('True', linewidth=0.5)
plt.rcParams.update({'font.size': 11})
plt.savefig('fig2_stdpr', dpi=300, edgecolor='w',bbox_inches="tight")
