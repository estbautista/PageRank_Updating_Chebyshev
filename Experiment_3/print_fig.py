import numpy as np
import scipy.stats as spt
import matplotlib.pyplot as plt

data_pr = np.load('Res_Fig3_stdPR.npz')

cheby_msgs = data_pr['cheby_msgs']
rwr_msgs = data_pr['rwr_msgs']
push_msgs = data_pr['push_msgs']

error_axis = data_pr['error_axis']
error_axis_push = data_pr['error_axis_push']

plt.figure(figsize=[7.5,2.5])
plt.xlim([1e-15,3e-4])
plt.ylim([-0.05e7,1.3e7])
plt.plot(error_axis, cheby_msgs, label='Cheby', linestyle='-', marker='o', color='tab:red', markersize=7, linewidth=2, markerfacecolor='None', markeredgewidth=2)
plt.plot(error_axis, rwr_msgs, label='RWR', linestyle='-', marker='^', color='tab:blue', markersize=7, linewidth=2, markerfacecolor='None', markeredgewidth=2)
plt.plot(error_axis_push, push_msgs, label='Push', linestyle='-', marker='s', color='tab:green', markersize=7, linewidth=2, markerfacecolor='None', markeredgewidth=2)
plt.legend(loc='upper center')
plt.ylabel('Messages', fontsize=13)
plt.xlabel('Error', fontsize=13)
plt.grid('True', linewidth=0.5)
plt.xscale('log')
plt.rcParams.update({'font.size': 11})
plt.savefig('fig3_stdpr', dpi=300, edgecolor='w',bbox_inches="tight")
