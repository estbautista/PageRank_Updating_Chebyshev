import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *
from static_algos import *
from updating_algos import *


filename = 'data/tech-as-topology.edges'
data = pd.read_csv(filename, delimiter=' ', header=None)
data = data.sort_values(3).to_numpy().astype(str)

# Process the data into a numeric form organized by snapshots
timestamps_dict, vertices_dict, snapshots_dict = process_data(data)

# Get the number of vertices
N = len(vertices_dict)

# Build matrices
init_graph = get_graph_object([snapshots_dict[0]], N)
evol_graph = get_graph_object([snapshots_dict[0], snapshots_dict[1]], N)

# Build the initial condition
np.random.seed(123)
seed = np.random.permutation(init_graph.active)[0]
init_graph.y = get_init_condition(seed, N)
evol_graph.y = get_init_condition(seed, N)

# Compute the exact PageRanks
alpha = 0.85
init_graph.exact_pr = PR_exact(init_graph, alpha)
evol_graph.exact_pr = PR_exact(evol_graph, alpha)

# Set the initial guess
evol_graph.init_guess = init_graph.exact_pr

# I change the target approximation error
target_error = [3e-15, 1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 3e-12, 1e-11, 3e-11,
				1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]

cheby_msgs, rwr_msgs, push_msgs, target_error_push = [], [], [], []
for error in target_error:
	# Update using Chebyshev
	evol_graph.apprx_cheby, messages_cheby = CP_update_error(init_graph, evol_graph, error, alpha)
	cheby_msgs.append( messages_cheby )
	
	# Update using Power Iteration
	evol_graph.apprx_rwr, messages_rwr = RWR_update_error(init_graph, evol_graph, error, alpha)
	rwr_msgs.append( messages_rwr )

	# Update using Gauss-Southwell
	if error >= 1e-6:
		evol_graph.apprx_push, messages_push = PUSH_update_error(init_graph, evol_graph, error, alpha)
		push_msgs.append( messages_push )
		target_error_push.append( error )

np.savez('Res_Fig3_stdPR', 
		cheby_msgs=cheby_msgs,
		rwr_msgs=rwr_msgs,
		push_msgs=push_msgs,
		error_axis=target_error,
		error_axis_push=target_error_push)

#plt.figure()
#plt.plot(target_error, cheby_msgs)
#plt.plot(target_error, rwr_msgs)
#plt.plot(target_error_push, push_msgs)
##plt.yscale('log')
#plt.xscale('log')
#plt.show()
