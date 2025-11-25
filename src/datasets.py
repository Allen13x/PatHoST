import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.neighbors import NearestNeighbors
from .spatial import numpy_to_torch_sparse
import scanpy as sc





def collate_fn(batch):
	"""
	Collates a list of samples into a batch for training or evaluation.
	This function takes a list of tuples, where each tuple represents a single sample
	containing input features (X), target labels (Y), and coordinates. It pads the
	input features and coordinates to the maximum sequence length in the batch to
	ensure uniform dimensions.
	Args:
		batch (list of tuples): A list where each element is a tuple (X, Y, coord).
			- X (torch.Tensor): Input features sequence.
			- Y (torch.Tensor): Target label.
			- coord (torch.Tensor): Coordinate sequence.
	Returns:
		tuple: A tuple containing the collated batch components:
			- X_batch (torch.Tensor): Padded input features with shape (batch_size, max_seq_len, ...).
			  Padded with -1.0.
			- Y_batch (torch.Tensor): Stacked target labels with shape (batch_size, ...).
			- coord_batch_padded (torch.Tensor): Padded coordinates with shape (batch_size, max_seq_len, ...).
			  Padded with 0.0 and cast to float32.
	"""
	X_batch, Y_batch, coord_batch = zip(*batch)
	X_batch = pad_sequence(X_batch, batch_first=True, padding_value=-1.0)
	Y_batch = torch.stack(Y_batch)
	coord_batch=[i.to(dtype=torch.float32) for i in coord_batch]
	
	coord_batch_padded = pad_sequence(coord_batch, batch_first=True, padding_value=0.0)
	return X_batch, Y_batch, coord_batch_padded

def create_mask(padded_sequences):
	"""
	Creates a binary mask for padded sequences to ignore padding tokens during attention mechanisms.

	This function identifies non-padding elements in the input sequences (assumed to be padded with 0.0)
	and generates a mask suitable for broadcasting in attention layers.

	Args:
		padded_sequences (torch.Tensor): A tensor containing padded sequences.
			Expected shape is usually (batch_size, sequence_length, feature_dim) or similar.

	Returns:
		torch.Tensor: A boolean or binary mask tensor.
			The shape is (batch_size, 1, 1, sequence_length) or (batch_size, 1, 1, sequence_length)
			depending on the input dimensions, designed to be broadcastable across attention heads.
	"""
	mask= (padded_sequences != 0.0).any(dim=-1)
	return mask.unsqueeze(1).unsqueeze(2)



class CustomDataset(Dataset):
	"""
	A custom PyTorch Dataset for loading paired data samples.
	This class wraps lists of input features and target labels, providing
	an interface to access them as PyTorch tensors.
	Args:
		X_list (list or array-like): A list or array containing the input features.
		Y_list (list or array-like): A list or array containing the target labels.
	Attributes:
		X_list (list or array-like): Stores the input features.
		Y_list (list or array-like): Stores the target labels.
	Methods:
		__len__(): Returns the total number of samples in the dataset.
		__getitem__(idx): Retrieves the sample and label at the specified index,
						  converted to float32 PyTorch tensors.
	"""
	def __init__(self, X_list, Y_list):
		self.X_list = X_list
		self.Y_list = Y_list

	def __len__(self):
		return len(self.X_list)

	def __getitem__(self, idx):
		X = self.X_list[idx]
		Y = self.Y_list[idx]
		return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def GetDataPoints(spdata,keys,Xdata='dataX',Ydata='alligned_Y',spots='overlapping_spots',signal_cutoff=0):
	"""
	Extracts data points (features and labels) from a spatial data dictionary based on specified keys.
	This function iterates through a list of keys to access specific datasets within `spdata`. It processes feature matrices (`Xdata`) and target labels (`Ydata`), applying a signal cutoff quantile to binarize the target labels. It handles cases where specific 'overlapping_spots' are defined or defaults to using all available data points if they are not.
	Args:
		spdata (dict): A dictionary containing spatial data objects. Each key maps to a dataset that should contain `Xdata`, `Ydata`, and optionally `spots`.
		keys (list): A list of keys (strings) representing the specific datasets within `spdata` to process.
		Xdata (str, optional): The key within the dataset to access the feature matrix. Defaults to 'dataX'.
		Ydata (str, optional): The key within the dataset to access the target label matrix. Defaults to 'alligned_Y'.
		spots (str, optional): The key within the dataset defining specific indices (overlapping spots) to extract. Defaults to 'overlapping_spots'.
		signal_cutoff (float, optional): A quantile value (between 0 and 1) used to threshold the target labels. Values below this quantile are set to 0, and the rest are binarized to 1. Defaults to 0.
	Returns:
		tuple: A tuple containing four elements:
			- X_list (list of np.ndarray): A list of feature arrays extracted from the datasets.
			- Y_list (list of int): A list of binarized target labels corresponding to the features.
			- KEYS (list of str): A list of keys indicating the source dataset for each extracted data point.
			- quantile (float): The calculated quantile value used for the last processed key's thresholding.
	"""
	Y_list=[]
	X_list=[]
	KEYS=[]
	for key in keys:
		y=spdata[key][Ydata].X.toarray()
		y=np.sum(y,axis=1)
		quantile=np.quantile(y,signal_cutoff)
		y[y<quantile]=0
		y=(y>0).astype(int)
		if spots in spdata[key]:
			for i,a in enumerate(range(len(spdata[key][spots]))):
				q=spdata[key][Xdata][spdata[key][spots][a]].X.toarray()
				q=q[:,spdata[key]['xcol']]
				X_list.append(q)
				y0=y[i]
				Y_list.append(y0)
				KEYS.append(key)
		else:
			for a in range(spdata[key][Xdata].X.shape[0]):
				q=spdata[key][Xdata][a].X.toarray()
				q=q[:,spdata[key]['xcol']]
				X_list.append(q)
				y0=y[a]
				Y_list.append(y0)
				KEYS.append(key)

	return X_list,Y_list,KEYS,quantile



def GetDataSets(spdata,keys,Xdata='dataX',Ydata='alligned_Y',spots='overlapping_spots',signal_cutoff=0, n_neigh=19):
	"""
	Extracts and processes datasets for Graph Neural Network training from spatial transcriptomics data.
	This function iterates through a dictionary of spatial data objects, constructs neighborhood graphs based on spatial coordinates, and prepares feature matrices (X), target labels (Y), and adjacency matrices (ADJ) for each spot or region of interest.
	Args:
		spdata (dict): A dictionary containing spatial data objects (e.g., AnnData) keyed by sample identifiers.
		keys (list): A list of keys (strings) representing the specific samples within `spdata` to process.
		Xdata (str, optional): The key within the spatial data object accessing the input feature data (e.g., gene expression). Defaults to 'dataX'.
		Ydata (str, optional): The key within the spatial data object accessing the target label data. Defaults to 'alligned_Y'.
		spots (str, optional): The key indicating a subset of specific spots to process (e.g., 'overlapping_spots'). If present, processing is restricted to these indices. Defaults to 'overlapping_spots'.
		signal_cutoff (float, optional): The quantile threshold (0 to 1) used to binarize the target variable Y. Values below this quantile are set to 0, others to 1. Defaults to 0.
		n_neigh (int, optional): The number of nearest neighbors to consider when constructing the spatial graph. Defaults to 19.
	Returns:
		tuple: A tuple containing the following elements:
			- X_list (list): A list of feature matrices (numpy arrays), where each element corresponds to the neighborhood of a spot.
			- Y_list (list): A list of target labels (integers), corresponding to the binarized signal of the central spot.
			- ADJ_list (list): A list of sparse adjacency matrices (torch sparse tensors), representing the local graph structure for each spot.
			- KEYS (list): A list of sample keys corresponding to each extracted data point.
			- quantile (float): The calculated signal threshold value used for the last processed key.
	"""
	Y_list=[]
	X_list=[]
	ADJ_list=[]
	KEYS=[]
	for key in keys:
		n_neigh=19
		if spots in spdata[key]:
			neigh=NearestNeighbors(n_neighbors=(n_neigh+1),metric='euclidean').fit(spdata[key][Xdata][spdata[key][spots]].obsm['spatial'])
			d,index=neigh.kneighbors(spdata[key][Xdata][spdata[key][spots]].obsm['spatial'])

		else:
			neigh=NearestNeighbors(n_neighbors=(n_neigh+1),metric='euclidean').fit(spdata[key][Xdata].obsm['spatial'])
			d,index=neigh.kneighbors(spdata[key][Xdata].obsm['spatial'])


		mean_d=np.mean(d[:,1:])

		dist_tot=mean_d + 20

		mean_d_first=np.mean(d[:,1:6])

		dist_first=mean_d_first + 20

		adj=np.zeros((len(index),len(index)))
		for i in range(len(index)):
			for j in range(1,n_neigh+1):
				if d[i][j]<dist_first:
					adj[i][index[i][j]]=1




		print(f"Mean distance between spots: {mean_d} for {key} - first neighbors {dist_first}, shape: {adj.shape}")



		y=spdata[key][Ydata].X.toarray()
		y=np.sum(y,axis=1)
		quantile=np.quantile(y,signal_cutoff)
		y[y<quantile]=0
		y=(y>0).astype(int)
		if spots in spdata[key]:
			for i,a in enumerate(range(len(spdata[key][spots]))):
				n_spots=index[a]
				idx = n_spots.astype(int)
				adj0 = adj[np.ix_(idx, idx)]	
				q=spdata[key][Xdata][spdata[key][spots]][n_spots].X.toarray()
				q=q[:,spdata[key]['xcol']]
				X_list.append(q)
				adj0=numpy_to_torch_sparse(adj0)
				ADJ_list.append(adj0)
				y0=y[i]
				Y_list.append(y0)
				KEYS.append(key)
		else:
			for a in range(spdata[key][Xdata].X.shape[0]):
				n_spots=index[a]
				idx = n_spots.astype(int)
				adj0 = adj[np.ix_(idx, idx)]
				q=spdata[key][Xdata][n_spots].X.toarray()
				q=q[:,spdata[key]['xcol']]
				X_list.append(q)
				adj0=numpy_to_torch_sparse(adj0)
				ADJ_list.append(adj0)
				y0=y[a]
				Y_list.append(y0)
				KEYS.append(key)

	return X_list,Y_list,ADJ_list,KEYS,quantile
