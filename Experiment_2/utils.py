import numpy as np
import scipy as sp
import scipy.sparse as sps

class graph_obj:
	def __init__(self):
		self.N = None
		self.A = None
		self.D = None
		self.L = None
		self.R = None
		self.y = None
		self.tmp_y = None
		self.active = None
		self.lmax = None

		self.exact_pr = None
		self.apprx_update_pr = None
		self.apprx_scratch_pr = None
		self.init_guess = None


def process_data(data):
	timestamps_dict = {}
	snapshots_dict = {}
	vertices_dict = {}
	for item in range(data.shape[0]):
		source, target, weight, timestamp = data[item]
		if timestamp not in timestamps_dict.keys(): timestamps_dict[timestamp] = len(timestamps_dict)
		if source not in vertices_dict.keys(): vertices_dict[source] = len(vertices_dict)
		if target not in vertices_dict.keys(): vertices_dict[target] = len(vertices_dict)
		if timestamps_dict[timestamp] not in snapshots_dict.keys(): snapshots_dict[timestamps_dict[timestamp]] = {'source':[], 'target':[]}
		
		snapshots_dict[timestamps_dict[timestamp]]['source'].append(vertices_dict[source])
		snapshots_dict[timestamps_dict[timestamp]]['target'].append(vertices_dict[target])
		snapshots_dict[timestamps_dict[timestamp]]['source'].append(vertices_dict[target])
		snapshots_dict[timestamps_dict[timestamp]]['target'].append(vertices_dict[source])
	
	return timestamps_dict, vertices_dict, snapshots_dict

def build_matrices(source_list, target_list, N):
	weights = np.ones(len(source_list))
	A = sps.coo_matrix((weights, (source_list, target_list)), shape=(N,N)).tocsr()
	A.sum_duplicates()
	A.data[:] = 1
	
	D = np.array(A.sum(axis=1)).squeeze()
	Deg = sps.diags([D], [0])
	Deg_inv = sps.diags([np.reciprocal(D, where=D!=0)], [0])
	L = Deg - A
	R = L.dot(Deg_inv)	

	graph = graph_obj()
	graph.N = N
	graph.A = A
	graph.D = D
	graph.R = R
	graph.L = L
	graph.lmax = 2
	graph.active = np.where(D > 0)[0]
	return graph

def get_graph_object(list_edgelists, N):
	
	source_list = []
	target_list = []
	for snapshot_data in list_edgelists:
		source_list += snapshot_data['source'] 
		target_list += snapshot_data['target']

	graph = build_matrices(source_list, target_list, N)
	return graph

def get_init_condition(seed, N):
	y = np.zeros(N)
	y[seed] = 1
	return y

def get_approximation_error(exact_pr, approx_pr):
	return np.linalg.norm(exact_pr - approx_pr)/np.linalg.norm(exact_pr)

def get_perturbation_size(graph_prev, graph_new, alpha):
	N = graph_new.N
	lmax = graph_new.lmax

	# Compute coefficients
	mu = (1-alpha)/alpha
	psi = -lmax/(2*mu + lmax)
	rho = (2*mu)/(2*mu + lmax)
	phi = 2/lmax

	# Operator difference
	diff_OP = graph_new.R - graph_prev.R

	# Compute the update distribution r 
	r = psi*phi*diff_OP.dot(graph_new.init_guess)
	r_norm = np.linalg.norm(r)
	return r_norm
