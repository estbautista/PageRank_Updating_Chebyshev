from static_algos import *
import numpy as np
import scipy as sp

def CP_update_error(graph_prev, graph_new, error, alpha):
	N = graph_new.N
	lmax = graph_new.lmax

	# Compute coefficients
	mu = (1-alpha)/alpha
	psi = -lmax/(2*mu + lmax)
	rho = (2*mu)/(2*mu + lmax)
	phi = 2/lmax

	# Difference operator
	diff_OP = graph_new.R - graph_prev.R

	# Nodes that changed trnsmit to their neighbors
	transmit_count = diff_OP.count_nonzero() 

	# Compute the update distribution r 
	graph_new.tmp_y = graph_new.y
	graph_new.y = psi*phi*diff_OP.dot(graph_new.init_guess)

	# Compute the update
	pr_approx, num_transm = CP_static_error(graph_new, error, alpha)
	graph_new.y = graph_new.tmp_y
	
	# Update the previous PageRank and the transmit count
	pr_update = graph_new.init_guess + pr_approx/rho
	transmit_count += num_transm

	return pr_update, transmit_count

def RWR_update_error(graph_prev, graph_new, error, alpha):
	N = graph_new.A.shape[0]

	# Difference operator
	diff_OP = graph_new.Pt - graph_prev.Pt
	
	# Nodes that changed transmit to their neighbors
	transmit_count = diff_OP.count_nonzero()

	# Compute the update distribution r
	graph_new.tmp_y = graph_new.y
	graph_new.y = alpha*diff_OP.dot(graph_new.init_guess)
	
	# Compute the update
	pr_approx, num_transm = RWR_static_error(graph_new, error, alpha)
	graph_new.y = graph_new.tmp_y

	# Update the previous PageRank and the transmit count
	pr_update = graph_new.init_guess + pr_approx/(1-alpha)
	transmit_count += num_transm

	return pr_update, transmit_count

def PUSH_update_error(graph_prev, graph_new, error, alpha):
	N = graph_prev.A.shape[0]

	# Difference operator
	diff_OP = graph_new.Pt - graph_prev.Pt

	# Nodes that changed transmit to their neighbors
	transmit_count = diff_OP.count_nonzero()

	# Compute the update distribution r
	graph_new.exact_up = (1-alpha)*(graph_new.exact_pr - graph_prev.exact_pr)
	r = alpha*diff_OP.dot(graph_new.init_guess)
	p = np.array(graph_new.init_guess)
	
	# Check approximation error
	apprx_error = get_approximation_error(graph_new.exact_pr, p)

	while apprx_error > error:
		# Push step
		ix_u = np.argmax(abs(r))
		r_u = r[ix_u]
		p[ix_u] += r_u
		r[ix_u] -= r_u

		indic_u = np.zeros(graph_new.N)
		indic_u[ix_u] = 1

		r = r + alpha*r_u*(graph_new.Pt.dot(indic_u))
		
		# Update the message count
		num_u_neighs = graph_new.D[ix_u]
		transmit_count += num_u_neighs

		# Update approximation error
		apprx_error = get_approximation_error(graph_new.exact_pr, p)

	return p, transmit_count

