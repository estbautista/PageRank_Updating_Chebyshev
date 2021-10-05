import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *
from static_algos import *
from updating_algos import *


filename = '../data/tech-as-topology.edges'
data = pd.read_csv(filename, delimiter=' ', header=None)
data = data.sort_values(3).to_numpy().astype(str)

# Process the data into a numeric form organized by snapshots
timestamps_dict, vertices_dict, snapshots_dict = process_data(data)

# Get the number of vertices
N = len(vertices_dict)

# Build matrices
init_graph = get_graph_object([snapshots_dict[0]], N)
evol_graph = get_graph_object([snapshots_dict[0], snapshots_dict[1]], N)

# Construct an initial condition
np.random.seed(123)
error_update_list, transmissions_update_list = [], []
error_scratch_list, transmissions_scratch_list = [], []

for realization in range(20):
	seed = np.random.permutation(init_graph.active)[0]
	print('seed = ', seed)
	init_graph.y = get_init_condition(seed, N)
	evol_graph.y = get_init_condition(seed, N)

	# Compute exact PageRanks
	alpha = 0.85
	print('Computing Exact PR Init')
	init_graph.exact_pr, _ = CP_static_iters(init_graph, 1300, alpha)
	print('Computing Exact PR Evolved')
	evol_graph.exact_pr, _ = CP_static_iters(evol_graph, 1300, alpha)
	evol_graph.init_guess = init_graph.exact_pr

	# Compute the approximation using Chebyshev polynomials
	error_update_realization, transmissions_update_realization = [], []
	error_scratch_realization, transmissions_scratch_realization = [], []

	for iters in range(10, 300, 20):
		print('Entering iter = ' + str(iters))
		# Get the new pagerank approximations
		evol_graph.apprx_update_pr, trans_update = CP_update_iters(init_graph, evol_graph, iters, alpha)
		evol_graph.apprx_scratch_pr, trans_scratch = CP_static_iters(evol_graph, iters, alpha)

		# Store the result
		error_update_realization.append( get_approximation_error(evol_graph.exact_pr, evol_graph.apprx_update_pr) )
		transmissions_update_realization.append( trans_update )

		error_scratch_realization.append( get_approximation_error(evol_graph.exact_pr, evol_graph.apprx_scratch_pr) )
		transmissions_scratch_realization.append( trans_scratch )
	
	error_update_list.append( error_update_realization )
	error_scratch_list.append( error_scratch_realization )
	transmissions_update_list.append( transmissions_update_realization )
	transmissions_scratch_list.append( transmissions_scratch_realization )
	
# Get mean values and std deviations
transmission_update_array = np.array( transmissions_update_list )
transmission_scratch_array = np.array( transmissions_scratch_list )
error_update_array = np.array( error_update_list )
error_scratch_array = np.array( error_scratch_list )

np.savez('Res_Fig1_gammaPR', 
		transmissions_update=transmission_update_array, 
		transmissions_scratch=transmission_scratch_array, 
		error_update=error_update_array, 
		error_scratch=error_scratch_array)


#plt.figure()
#plt.errorbar(np.mean(transmission_update_array, axis=0), np.mean(error_update_array, axis=0), \
#			xerr=np.std(transmission_update_array, axis=0), yerr=np.std(error_update_array, axis=0), label='Upate', linestyle='None', marker='o')
#plt.errorbar(np.mean(transmission_scratch_array, axis=0), np.mean(error_scratch_array, axis=0), \
#			xerr=np.std(transmission_scratch_array, axis=0), yerr=np.std(error_scratch_array, axis=0), label='Scratch', linestyle='None', marker='^')
#
#plt.legend()
#plt.yscale('log')
#plt.show()
