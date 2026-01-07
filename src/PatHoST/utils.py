from joblib import dump, load
import numpy as np
import scipy.sparse as sp
import torch



def merge_nested_dicts(d1, d2):
	"""
	Recursively merges two dictionaries.
	If a key exists only in one of the dictionaries, it is added to the result.
	If a key exists in both and the value is a dictionary, the merge is performed recursively.
	If a key exists in both and the value is NOT a dictionary, the value in d1 is overwritten with the value in d2.
	"""
	for key, value in d2.items():
		if key in d1:
			if isinstance(d1[key], dict) and isinstance(value, dict):
				merge_nested_dicts(d1[key], value)

				# If you want to keep both pieces of information in case of a conflict,
				# you could, for example, concatenate the values into a list:
				# d1[key] = [d1[key], value] or decide on another logic.
				# Here we choose to overwrite with the value from d2:
				d1[key] = value
		else:
			d1[key] = value
	return d1



def load_nested_keys_on_demand(outer_keys, inner_keys, path='.'):
	"""
	Load specific keys from nested dictionaries stored in pickle files.

	Args:
		outer_keys (list): List of outer keys corresponding to pickle file names.
		inner_keys (list): List of inner keys to extract from each loaded dictionary.
		path (str): Path to the directory containing the pickle files.

	Returns:
		dict: A dictionary where each outer key maps to a dictionary of the requested inner keys.
	"""
	result = {}
	for outer_key in outer_keys:
		try:
			inner_dict = load(f'{path}/{outer_key}.pkl')
			result[outer_key] = {inner_key: inner_dict[inner_key] for inner_key in inner_keys if inner_key in inner_dict}
		except FileNotFoundError:
			print(f"File for {outer_key} not found.")
		except Exception as e:
			print(f"Error loading {outer_key}: {e}")
	return result

def normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)


def numpy_to_torch_sparse(adj_numpy):
	"""
	Converts a NumPy adjacency matrix into a PyTorch sparse tensor.

	This function identifies the non-zero elements in the input NumPy array to create
	the indices and values required for a PyTorch sparse FloatTensor. It assumes an
	unweighted adjacency matrix where the existence of an edge implies a value of 1.

	Args:
		adj_numpy (numpy.ndarray): The input adjacency matrix as a dense NumPy array.

	Returns:
		torch.sparse.FloatTensor: A sparse PyTorch tensor representation of the adjacency matrix,
		where indices correspond to non-zero elements in the input and values are all ones.
	"""
	row, col = np.nonzero(adj_numpy)
	indices = torch.tensor(np.vstack((row, col)), dtype=torch.long)
	values = torch.ones(len(row))
	shape = torch.Size(adj_numpy.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

class EarlyStopping:
	"""Early stopping ottimizzato con warmup e patience dinamica"""
	def __init__(self, patience=5, min_delta=1e-6, warmup_epochs=10):
		self.patience = patience
		self.min_delta = min_delta
		self.warmup_epochs = warmup_epochs
		self.counter = 0
		self.best_loss = float('inf')
		self.epoch_count = 0
	
	def __call__(self, val_loss):
		self.epoch_count += 1
		

		if self.epoch_count <= self.warmup_epochs:
			if val_loss < self.best_loss:
				self.best_loss = val_loss
			return False
		

		if val_loss < self.best_loss - self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
			return False
		else:
			self.counter += 1
			return self.counter >= self.patience