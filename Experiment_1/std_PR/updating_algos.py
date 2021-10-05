from static_algos import *
import numpy as np
import scipy as sp

def CP_update_iters(graph_prev, graph_new, iters, alpha):
	N = graph_new.N
	lmax = graph_new.lmax

	# Compute coefficients
	mu = (1-alpha)/alpha
	psi = -lmax/(2*mu + lmax)
	rho = (2*mu)/(2*mu + lmax)
	phi = 2/lmax

	# Previous pr
	diff_OP = graph_new.R - graph_prev.R

	# Nodes that changed transmit to their neighbors
	transmit_count = diff_OP.count_nonzero()

	# Compute the update distribution r 
	graph_new.tmp_y = graph_new.y
	graph_new.y = psi*phi*diff_OP.dot(graph_new.init_guess)

	# Compute the update
	pr_approx, num_transm = CP_static_iters(graph_new, iters, alpha)
	graph_new.y = graph_new.tmp_y
	
	# Update the previous PageRank and the transmit count
	pr_update = graph_new.init_guess + pr_approx/rho
	transmit_count += num_transm

	return pr_update, transmit_count
