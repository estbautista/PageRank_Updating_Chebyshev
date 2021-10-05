import numpy as np
import scipy.stats as spt
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = data.shape[0]
	m, se = np.mean(a, axis=0), spt.stats.sem(a, axis=0)
	return m, se

data_pr = np.load('Res_Fig1_stdPR.npz')

mean_err_up, bar_err_up = mean_confidence_interval(data_pr['error_update'])
mean_err_sc, bar_err_sc = mean_confidence_interval(data_pr['error_scratch'])
mean_trs_up, bar_trs_up = mean_confidence_interval(data_pr['transmissions_update'])
mean_trs_sc, bar_trs_sc = mean_confidence_interval(data_pr['transmissions_scratch'])

#plt.figure(figsize=[3,2.5])
plt.figure(figsize=[7.5, 2.5])
plt.errorbar(mean_trs_up, mean_err_up, xerr=bar_trs_up, yerr=bar_err_up, label='Update', linestyle='None', marker='o', color='tab:red', markersize=10, linewidth=3, markerfacecolor='None', markeredgewidth=1.5)
plt.errorbar(mean_trs_sc, mean_err_sc, xerr=bar_trs_sc, yerr=bar_err_sc, label='Scratch', linestyle='None', marker='^', color='tab:blue', markersize=10, linewidth=3, markerfacecolor='None', markeredgewidth=1.5)
plt.legend()
plt.ylabel('Error', fontsize=13)
plt.xlabel('Messages', fontsize=13)
#plt.title('Std. PageRank')
plt.grid('True', linewidth=0.5)
plt.yscale('log')
plt.rcParams.update({'font.size': 11})
plt.savefig('fig1_stdpr', dpi=300, edgecolor='w',bbox_inches="tight")
