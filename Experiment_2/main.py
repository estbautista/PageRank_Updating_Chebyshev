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
print(len(snapshots_dict))

# Get the number of vertices
N = len(vertices_dict)

# Build the initial graph
init_graph = get_graph_object([snapshots_dict[0]], N)

# Set the seed for the initial condition
np.random.seed(1234)
seed = np.random.permutation(init_graph.active)[0]
init_graph.y = get_init_condition(seed, N)

# Compute the exact PageRank for the initial graph
threshold = 1e-13
alpha = 0.85
init_graph.exact_pr = PR_exact(init_graph, alpha)

transmission_update_list = []
transmission_scratch_list = []
perturbation_list = []
#test_vals = [10, 25, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4300, 4600]
test_vals = [1, 5, 10, 20, 30, 40, 60, 80, 100, 150, 
			200, 250, 300, 400, 500, 600, 750,
			900, 1100, 1300, 1500, 1700, 1900, 2100, 2300,
			2500, 2800, 3100, 3400, 3700, 4000, 4300, 4600]
for snapshot in test_vals:
	new_edges = []
	for time in range(snapshot + 1):
		new_edges.append(snapshots_dict[time])
	
	# Build evolved graph
	evol_graph = get_graph_object(new_edges, N)
	evol_graph.y = get_init_condition(seed, N)
	evol_graph.init_guess = init_graph.exact_pr
	
	# Coompute the exact PageRank 
	evol_graph.exact_pr = PR_exact(evol_graph, alpha)

	# Get the norm of the update distribution
	r_norm = get_perturbation_size(init_graph, evol_graph, alpha)
	
	# Reference: count how many transmissions are needed from scratch to attain the threshold. 
	err = 1
	iters = 2
	while err > threshold:
		evol_graph.apprx_scratch_pr, trans_scratch = CP_static_iters(evol_graph, iters, alpha)
		err = get_approximation_error( evol_graph.exact_pr, evol_graph.apprx_scratch_pr )
		iters += 1
	print('Iters Scratch = ', iters)

	# Update until the error is below a threshold
	err = 1
	iters = 2
	while err > threshold:
		evol_graph.apprx_update_pr, trans_update = CP_update_iters(init_graph, evol_graph, iters, alpha)
		err = get_approximation_error( evol_graph.exact_pr, evol_graph.apprx_update_pr )
		iters += 1

	print('Iters Update = ', iters)
	transmission_update_list.append(trans_update)
	transmission_scratch_list.append(trans_scratch)
	perturbation_list.append( evol_graph.A.count_nonzero() - init_graph.A.count_nonzero() )

np.savez('Res_Fig2_stdPR', 
		transmissions_update=transmission_update_list, 
		transmissions_scratch=transmission_scratch_list, 
		perturbation_size=perturbation_list)


#plt.figure()
#plt.plot(perturbation_list, transmission_update_list, label='Update', marker='o', linestyle='None')
#plt.plot(perturbation_list, transmission_scratch_list, label='Scratch', marker='^', linestyle='None')
##plt.xscale('log')
#plt.show()
