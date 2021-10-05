import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
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
evol_edgelist = []
for k in range(100): evol_edgelist.append( snapshots_dict[k] )
init_graph = get_graph_object(evol_edgelist, N)

# Build initial condition
np.random.seed(123)
seed = np.random.permutation(init_graph.active)[0]
init_graph.y = get_init_condition(seed, N)

# Compute the exact PageRank
alpha = 0.85
init_graph.exact_pr = PR_exact(init_graph, alpha)
init_graph.approx_update_pr = init_graph.exact_pr

# We let the graph evolve
perturbation_size = []
error_update, error_scratch = [], []
for snapshot in range(100, 1100):
	
	# Update the edgelist
	perturbation_size.append( len(snapshots_dict[snapshot]['source']) )
	evol_edgelist.append( snapshots_dict[snapshot] )

	# Build the evolved graph
	evol_graph = get_graph_object(evol_edgelist, N)
	evol_graph.y = get_init_condition(seed, N)

	# Compute the exact PageRank of evolved graph
	evol_graph.exact_pr = PR_exact(evol_graph, alpha)

	# Set the previous PR as the initial guess of the evolved graph
	evol_graph.init_guess = init_graph.approx_update_pr

	# Perform the update. Use 50 iterations
	evol_graph.approx_update_pr, _ = CP_update_iters(init_graph, evol_graph, 30, alpha)

	# Compute new pagerank from scratch using the same iterations
	evol_graph.approx_scratch_pr, _ = CP_static_iters(evol_graph, 30, alpha)

	# Asess the error
	err_update = get_approximation_error( evol_graph.exact_pr, evol_graph.approx_update_pr )
	err_scratch = get_approximation_error( evol_graph.exact_pr, evol_graph.approx_scratch_pr )

	# Store result
	error_update.append( err_update )
	error_scratch.append( err_scratch )

	# Re-assign the graphs
	init_graph = copy.deepcopy(evol_graph)
	
	print('-----------')
	print('Snapshot = ', snapshot)
	print('Error update = ', err_update)
	print('Error scratch = ', err_scratch)

np.savez('Res_Fig4_stdPR', 
		error_update=error_update,
		error_scratch=error_scratch,
		perturbation_size=perturbation_size)
