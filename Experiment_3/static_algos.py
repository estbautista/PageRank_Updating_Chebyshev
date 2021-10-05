import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from utils import *

def PR_exact(graph, alpha):
	mu = (1-alpha)/alpha
	Identity = sps.eye(graph.N)
	OP = graph.R + mu*Identity
	f = spl.spsolve(OP, graph.y)
	return mu*f

def update_transmit_count(graph, vec):	
	# We transmit vec to the neighbors. Only nodes with nonzero signal transmit
	nnz_idx = np.where(vec != 0)[0]
	neighbors = graph.D[nnz_idx]
	return sum(neighbors)

def CP_static_error(graph, error, alpha):
	# Track number of transmissions
	transmit_count = 0

	# Parameters
	N = graph.N
	mu = (1-alpha)/alpha
	theta = np.linspace(0,np.pi,30000)
	step = theta[1] - theta[0]
	lmax = graph.lmax
	rho = (2*mu)/(2*mu + lmax)

	# Graph and initial condition
	OP = (2/lmax)*graph.R - sps.eye(N)
	y = np.array(graph.y)

	# coefficients
	filt_arg = (lmax/2)*(np.cos(theta) + 1)
	filt = np.divide(mu, mu + filt_arg)
	tmp1 = np.multiply(np.cos(0*theta),filt*step); tmp1[0]=tmp1[0]/2; tmp1[-1]=tmp1[-1]/2;
	tmp2 = np.multiply(np.cos(1*theta),filt*step); tmp2[0]=tmp2[0]/2; tmp2[-1]=tmp2[-1]/2;
	coef1 = (2/np.pi)*np.sum(tmp1)
	coef2 = (2/np.pi)*np.sum(tmp2)

	# Polynomial elements
	transmit_count += update_transmit_count(graph, y)
	polyTerm_2back = np.array(y)
	polyTerm_1back = OP.dot(y)

	# Chebyshev approximation
	Cheby_approximation_prev = 0.5*coef1*polyTerm_2back + coef2*polyTerm_1back;
	Cheby_approximation_curr = np.array(Cheby_approximation_prev)

	it = 2;
	apprx_error = get_approximation_error(graph.exact_pr, graph.init_guess + (Cheby_approximation_curr/rho))
	while apprx_error > error:
		# Chebyshev coefficient
		tmp = np.array(np.multiply(np.cos(it*theta),filt*step)); tmp[0]=tmp[0]/2; tmp[-1]=tmp[-1]/2;
		coef_curr = (2/np.pi)*np.sum(tmp);

		# Current polynomial term
		transmit_count += update_transmit_count(graph, polyTerm_1back)
		polyTerm_curr = (2*OP).dot(polyTerm_1back) - polyTerm_2back;

		# Chebyshev approximation
		Cheby_approximation_curr = np.array(Cheby_approximation_prev + coef_curr*polyTerm_curr);

		# Update
		apprx_error = get_approximation_error(graph.exact_pr, graph.init_guess + (Cheby_approximation_curr/rho))
		polyTerm_2back = np.array(polyTerm_1back).reshape(N);
		polyTerm_1back = np.array(polyTerm_curr).reshape(N);
		Cheby_approximation_prev = np.array(Cheby_approximation_curr);
		it += 1
		
	return Cheby_approximation_curr, transmit_count

def RWR_static_error(graph, error, alpha):

	# Track number of transmissions
	transmit_count = 0

	# Parameters
	N = graph.A.shape[0]
	
	# compute initial distribution
	p = np.array(graph.y);

	apprx_error = get_approximation_error(graph.exact_pr, graph.init_guess + (p/(1-alpha)))
	while apprx_error > error:
		# Update the number of transmissions
		transmit_count += update_transmit_count(graph, p)

		# next iteration in approx
		p = (1 - alpha)*graph.y + alpha*graph.Pt.dot(p) 

		# Compute error
		apprx_error = get_approximation_error(graph.exact_pr, graph.init_guess + (p/(1-alpha)))

	return p, transmit_count
