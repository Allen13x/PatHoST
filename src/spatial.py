import math
from math import exp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.linalg import svd
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import scipy.sparse as sp

def LogNormSpData(spdata, key, slot='dataX'):
	"""
	Perform log normalization on spatial data.

	Parameters:
	spdata (dict): Dictionary containing spatial data.
	key (str): Key to access specific data in spdata.
	slot (str): Slot name to access the data matrix. Default is 'dataX'.

	Returns:
	AnnData: Log-normalized AnnData object.
	"""
	d = spdata[key][slot].copy()
	sc.pp.normalize_total(d, inplace=True)
	sc.pp.log1p(d)
	return d


def ProjectBacterialGT(spdata, key, GT, k=1, t=300):
	"""
	Aligns and projects bacterial ground truth (GT) spatial data onto a reference spatial dataset.
	This function takes two spatial datasets, aligns them using an optimal rotation matrix, and 
	projects the ground truth data onto the reference dataset. It calculates overlaps between 
	the transformed ground truth and the reference spatial data based on a distance threshold.
	Parameters:
	-----------
	spdata : dict
		A dictionary containing spatial data. Each key corresponds to a dataset, and each dataset 
		contains attributes such as 'dataX' (spatial data) and 'ycol' (column indices for features).
	key : str
		The key in `spdata` corresponding to the reference dataset.
	GT : str
		The key in `spdata` corresponding to the ground truth dataset to be projected.
	k : float, optional, default=1
		A blending factor between 0 and 1. Determines the weight of the ground truth data in the 
		final projection. `k=0` uses only the reference data, while `k=1` uses only the ground truth data.
	t : float, optional, default=300
		Distance threshold for determining overlaps between the transformed ground truth and the 
		reference spatial data.
	Returns:
	--------
	dummy : AnnData
		The modified reference dataset with blended features based on the ground truth data.
	overlapping_indices_x : list of int
		Indices of the reference dataset that overlap with the ground truth data.
	overlapping_indices_GT : list of int
		Indices of the ground truth dataset that overlap with the reference data.
	transformed_GT : ndarray
		The transformed ground truth spatial coordinates after alignment.
	nearest_indices : ndarray
		Indices of the nearest ground truth points for each reference point.
	distances : ndarray
		Pairwise distances between the reference spatial data and the transformed ground truth data.
	Raises:
	-------
	AssertionError
		If `k` is not between 0 and 1.
	Notes:
	------
	- The alignment is performed using Singular Value Decomposition (SVD) to compute the optimal 
		rotation matrix.
	- The function assumes that the input datasets contain spatial coordinates in the `obsm['spatial']` 
		attribute and metadata in the `obs` attribute.
	"""
	assert k >= 0 and k <= 1, "k must be between 0 and 1"
	anchor_x=spdata[key]['dataX'][spdata[key]['dataX'][spdata[key]['dataX'].obs['type']!='no_anchors'].obs.sort_values(by='type').index].obsm['spatial']
	anchor_name=spdata[key]['dataX'][spdata[key]['dataX'][spdata[key]['dataX'].obs['type']!='no_anchors'].obs.sort_values(by='type').index].obs['type']
	anchor_GT=spdata[GT]['dataX'][spdata[GT]['dataX'][spdata[GT]['dataX'].obs['type'].isin(anchor_name)].obs.sort_values(by='type').index].obsm['spatial']
	

	# Center the matrices
	anchor_x_centered = anchor_x - np.mean(anchor_x, axis=0)
	anchor_GT_centered = anchor_GT - np.mean(anchor_GT, axis=0)

	# Compute the optimal rotation matrix using SVD
	U, _, Vt = svd(np.dot(anchor_GT_centered.T, anchor_x_centered))
	R = np.dot(U, Vt)

	transformed_GT = np.dot(spdata[GT]['dataX'].obsm['spatial'] - np.mean(anchor_GT, axis=0), R) + np.mean(anchor_x, axis=0)

	# Calculate pairwise distances between transformed_b and anchor_c
	distances = cdist( spdata[key]['dataX'].obsm['spatial'],transformed_GT)

	# Define a distance threshold for overlap (e.g., 1e-5)
	threshold = t
	nearest_indices = np.argmin(distances, axis=1)

	overlaps = [(i, j) for i, j in enumerate(nearest_indices) if distances[i, j] < threshold]
	# Extract the indices of transformed_b that overlap with zdata.obsm['spatial']
	overlapping_indices_GT = [j for _, j in overlaps]
	overlapping_indices_x = [i for i, _ in overlaps]


	sorted_indices_GT=overlapping_indices_GT
	# Reorder transformed_b based on the sorted indices in reverse order

	dummy=spdata[key]['dataX'][overlapping_indices_x].copy()
	dummy=dummy[:, spdata[key]['ycol']]

	b= spdata[GT]['dataX'].X[sorted_indices_GT]
	b=b[:, spdata[GT]['ycol']]

	dummy.X=(1-k)*dummy.X + k*b

	return dummy, overlapping_indices_x, overlapping_indices_GT, transformed_GT,nearest_indices,distances




def ComputeGTonZaxis(spdata, key, GTs, zaxis, z0possition, zcoeff=0.04, z0coeff=0):
	"""
	Compute a weighted combination of gene expression data along the Z-axis.
	This function processes spatial gene expression data and computes a weighted 
	combination of gene expression signals based on their positions along the Z-axis. 
	The weights are determined by the distance from a reference Z position and 
	a specified coefficient.
	Args:
		spdata (dict): A dictionary containing spatial data. It is expected to have 
			keys like 'dataX' and 'ycol' for accessing the data matrix and column indices.
		key (str): The key in `spdata` to access the relevant data.
		GTs (list): A list of gene expression data (ground truth) to be projected.
		zaxis (list): A list of Z-axis positions corresponding to the GTs.
		z0possition (float): The reference Z-axis position.
		zcoeff (float, optional): Coefficient for weighting based on Z-axis distance. 
			Default is 0.04.
		z0coeff (float, optional): Coefficient for scaling the initial data. Default is 0.
	Returns:
		tuple:
			- dummy (AnnData): The resulting AnnData object containing the weighted 
				combination of gene expression data.
			- final_overlapping_spots (list): A list of spots that are common across 
				all GTs and used in the computation.
	Raises:
		AssertionError: If the number of GTs does not match the number of Z-axis positions.
	Notes:
		- The function assumes that the `ProjectBacterialGT` function is available 
			and used to project the GTs onto the spatial data.
		- The Z-axis weights are computed using an exponential decay function based 
			on the `zcoeff` parameter.
		- Only spots that are common across all GTs are considered in the final computation.
	Example:
		dummy, overlapping_spots = ComputeGTonZaxis(
			spdata=spatial_data,
			key="sample_key",
			GTs=gene_truths,
			zaxis=z_positions,
			z0possition=0.0,
			zcoeff=0.05,
			z0coeff=0.1
		)
	"""

	assert len(GTs)==len(zaxis), "Number of GTs must be equal to number of zaxis positions"

	zaxis=[x - z0possition for x in zaxis]



	gtdata=[]
	x_spots=[]

	for i,GT in enumerate(GTs):
		gtdata_i,x_spots_i,_,_,_,_=ProjectBacterialGT(spdata,key,GT,k=1,t=300)
		gtdata.append(gtdata_i)
		x_spots.append(x_spots_i)
	

	gt_pos=zip(gtdata,zaxis)



	if x_spots:
		inner_join=set(x_spots[0])
		for spots in x_spots[1:]:
			inner_join=inner_join.intersection(spots)
	else:
		inner_join = set()


	final_overlapping_spots=list(inner_join)

	index_mapping=[]
	for i in range(len(x_spots)):

		index_map = {spot: idx for idx, spot in enumerate(x_spots[i]) if spot in final_overlapping_spots}
		index_mapping.append(index_map)

	dummy=spdata[key]['dataX'][final_overlapping_spots].copy()
	dummy=dummy[:, spdata[key]['ycol']]
	dummy.X=z0coeff*dummy.X
	for i,(gtdata,z) in enumerate(gt_pos):
		z=exp(-zcoeff*abs(z))
		print(f"GT {i} Z score: {z}")
		indexes=[index_mapping[i][spot] for spot in final_overlapping_spots]
		dd = gtdata[indexes]
		dummy.X=dummy.X + z*dd.X
	
	return dummy, final_overlapping_spots



def visualize_signal_transformed_GT(spdata, key='Train_2', GT='GT_1', t=300):
	"""
	Visualize the transformed ground truth (GT) signal and its spatial relationship 
	with the projected signal data.
	Parameters:
	-----------
	spdata : dict
		A dictionary containing spatial data and associated metadata. It is expected 
		to have keys corresponding to `key` and `GT`, where `spdata[key]` contains 
		the projected data and `spdata[GT]` contains the ground truth data.
	key : str, optional
		The key in `spdata` representing the projected data. Default is 'Train_2'.
	GT : str, optional
		The key in `spdata` representing the ground truth data. Default is 'GT_1'.
	t : int, optional
		A parameter passed to the `ProjectBacterialGT` function, likely controlling 
		the number of nearest neighbors or a threshold. Default is 300.
	Returns:
	--------
	None
		This function generates a visualization with four subplots:
		- Top-left: Projected signal data and transformed GT data.
		- Top-right: Transformed GT data and projected signal data.
		- Bottom-left: Projected signal data with color indicating distance to the 
			nearest GT spot.
		- Bottom-right: Transformed GT data with color indicating distance to the 
			nearest projected spot.
	Notes:
	------
	- The function uses `ProjectBacterialGT` to compute the transformed GT data, 
		overlapping indices, and distances.
	- The visualization includes scatter plots with color-coded distances and 
		spatial relationships between the projected and GT data.
	- The function assumes that `spdata` contains the necessary data structures 
		and attributes, such as `obsm['spatial']` and `X.toarray()`.
	"""

	d,overlapping_indices_x, overlapping_indices_GT,transformed_GT,n,distances=ProjectBacterialGT(spdata,key,'GT_1',k=1,t=t)
	coords_x = d.obsm['spatial']
	coords_transformed_GT = transformed_GT
	Bact_sign_projected=(np.sum(d.X.toarray(),axis=1)>0).astype(int)
	Bact_sign_GT_projected=(np.sum(spdata[GT]['dataX'].X[:,spdata[GT]['ycol']].toarray(),axis=1)>0).astype(int)
	

	all_indices_GT = set(range(spdata[GT]['dataX'].shape[0]))
	unassigned_indices_GT = list(all_indices_GT - set(overlapping_indices_GT))
	D=[distances[i,j] for (i,j) in enumerate(n)]
	D=[D[i] for i in overlapping_indices_x]
	D=[float(d) for d in D]

	nearest_indices = np.argmin(distances, axis=0)
	D_GT = [distances[i, j] for j, i in enumerate(nearest_indices)]
	D_GT = [float(d) for d in D_GT]

	fig,axs= plt.subplots(2,2,figsize=(20,20))
	axs[0,0].scatter(coords_x[:, 0], coords_x[:, 1], c='grey', label=f'{key} Projected', alpha=1)
	axs[0,0].scatter(coords_transformed_GT[:, 0], coords_transformed_GT[:, 1], c=Bact_sign_GT_projected,cmap='viridis', label='GT Projected', alpha=0.5)

	axs[0,1].scatter(coords_transformed_GT[:, 0], coords_transformed_GT[:, 1], c='grey', label=f'{key} Projected', alpha=1)
	axs[0,1].scatter(coords_x[:, 0], coords_x[:, 1], c=Bact_sign_projected,cmap='viridis', label='GT Projected', alpha=0.5)

	scatter = axs[1,0].scatter(coords_x[:, 0], coords_x[:, 1], c=D, cmap='viridis', label=f'{key} Projected', alpha=1)
	colorbar = plt.colorbar(scatter, ax=axs[1,0])
	colorbar.set_label('Distance to Nearest GT spot')

	scatter_GT = axs[1,1].scatter(coords_transformed_GT[:, 0], coords_transformed_GT[:, 1], c=D_GT, cmap='viridis', label='GT Projected', alpha=1,vmax=300)
	colorbar_GT = plt.colorbar(scatter_GT, ax=axs[1,1])
	colorbar_GT.set_label('Distance to Nearest x Spot')

	plt.show()



def GetLocalPCADistance(spdata,key,X,Dist=200):
	"""
	Calculates a distance matrix based on local PCA similarity within spatial neighborhoods.
	This function identifies spatial neighbors for each observation, performs PCA on the 
	subset of data corresponding to these neighbors, and computes a weighted distance 
	metric based on the cosine similarity of the PCA components.
	Parameters
	----------
	spdata : dict
		A dictionary containing spatial data objects (likely AnnData objects).
	key : str
		The key to access the specific dataset within `spdata`.
	X : str or int
		The key or index to access the specific sample/layer within `spdata[key]`.
	Dist : float, optional
		The spatial distance threshold for including neighbors in the local PCA calculation. 
		Neighbors further than this Euclidean distance are excluded. Default is 200.
	Returns
	-------
	numpy.ndarray
		A square matrix (N x N) representing the calculated distances or weights between 
		observations, where N is the number of observations in `spdata[key][X]`. 
		The matrix is sparse, with non-zero values only for spatially local neighbors.
	"""
	n_neigh=50

	# Calculate spatial neighbors
	neigh=NearestNeighbors(n_neighbors=(n_neigh+1),metric='euclidean').fit(spdata[key][X].obsm['spatial'])
	dist,index=neigh.kneighbors(spdata[key][X].obsm['spatial'])
	D=np.zeros([spdata[key][X].shape[0],spdata[key][X].shape[0]])

	d=[1]*(n_neigh+1)

	for w in range(index.shape[0]):
		i=index[w][dist[w]<Dist]
		l=len(i)
		t=spdata[key][X][:,spdata[key]['xcol']][i].copy()
		sc.pp.pca(t)

		d=[0]*(l)
		for z in range(1,(l)):
			a=math.exp(2-distance.cosine(t.obsm['X_pca'][0],t.obsm['X_pca'][z]))
			d[z]=a
		d=[a/sum(d) for a in d]
		for q in range(0,l):
			D[i[0],i[q]]=d[q]
	
	return D



def GetGlobalPCADistance(spdata,key,X,Dist=200):
	"""
	Calculates a global distance matrix based on PCA similarity within spatial neighborhoods.
	This function computes a weighted adjacency matrix where weights represent the similarity
	between a cell and its spatial neighbors. Similarity is derived from the cosine distance
	of PCA representations of a specific feature subset (defined by 'xcol'), weighted exponentially.
	Parameters
	----------
	spdata : dict
		A dictionary containing spatial data objects (likely AnnData objects).
	key : str
		The key to access the specific dataset within `spdata`.
	X : str or int
		The key or index to access the specific sample/replicate within `spdata[key]`.
	Dist : float, optional
		The spatial distance threshold for defining valid neighbors. Neighbors further than
		this Euclidean distance are excluded from the calculation. Default is 200.
	Returns
	-------
	numpy.ndarray
		A square matrix (N x N, where N is the number of observations) representing the
		calculated global PCA distances/weights between cells. The matrix is sparse,
		populated only for spatially local neighbors.
	Notes
	-----
	- The function hardcodes `n_neigh=50` for the initial nearest neighbor search.
	- It performs PCA on a subset of data defined by `spdata[key]['xcol']`.
	- The weights are calculated using `exp(2 - cosine_distance)` and then normalized
	  so that the weights for a given cell's neighbors sum to 1 (though the normalization
	  logic appears to normalize based on the sum of the unnormalized weights).
	"""

	n_neigh=50

	# Calculate spatial neighbors
	neigh=NearestNeighbors(n_neighbors=(n_neigh+1),metric='euclidean').fit(spdata[key][X].obsm['spatial'])
	dist,index=neigh.kneighbors(spdata[key][X].obsm['spatial'])
	D=np.zeros([spdata[key][X].shape[0],spdata[key][X].shape[0]])

	d=[1]*(n_neigh+1)
	dummy=spdata[key][X][:,spdata[key]['xcol']].copy()
	sc.pp.pca(dummy)
	for w in range(index.shape[0]):
		i=index[w][dist[w]<Dist]
		l=len(i)
		t=dummy[i]

		d=[0]*(l)
		for z in range(1,(l)):
			a=math.exp(2-distance.cosine(t.obsm['X_pca'][0],t.obsm['X_pca'][z]))
			d[z]=a
		d=[a/sum(d) for a in d]
		for q in range(0,l):
			D[i[0],i[q]]=d[q]
	return D



def Smooth(spdata,key,data,X,overlap=None,mode='Mean',Dist=200, D=None):
	"""
	Smooths spatial data based on neighbor relationships using different modes.
	This function applies smoothing to a specific data layer within a spatial data object (`spdata`).
	It supports three modes: 'Mean', 'Local', and 'Global', which determine how the smoothing matrix `D`
	is calculated or utilized.
	Parameters
	----------
	spdata : dict
		A dictionary containing spatial data objects (e.g., AnnData objects).
	key : str
		The key in `spdata` identifying the specific sample or dataset to process.
	data : str
		The key within `spdata[key]` representing the source data layer to be smoothed (e.g., raw counts).
	X : str
		The key within `spdata[key]` representing the layer used for spatial coordinates or embeddings
		(e.g., where `.obsm['spatial']` is stored).
	overlap : list or None, optional
		A list of indices or a key in `spdata[key]` defining a subset of observations to process.
		If None, all observations in `spdata[key][X]` are used. Default is None.
	mode : str, optional
		The smoothing mode. Options are:
		- 'Mean': Calculates a nearest-neighbor graph based on Euclidean distance of spatial coordinates
		  and averages neighbors within a specific distance (`Dist`).
		- 'Local': Uses a pre-calculated or newly calculated local PCA distance matrix.
		- 'Global': Uses a pre-calculated or newly calculated global PCA distance matrix.
		Default is 'Mean'.
	Dist : float, optional
		The distance threshold for defining neighbors.
		- In 'Mean' mode: The Euclidean distance cutoff.
		- In 'Local'/'Global' modes: Passed to the respective distance calculation functions.
		Default is 200.
	D : ndarray or None, optional
		A pre-computed distance/weight matrix. If None and mode is 'Local' or 'Global',
		the matrix is calculated using helper functions (`GetLocalPCADistance` or `GetGlobalPCADistance`).
		Default is None.
	Returns
	-------
	dummy : AnnData
		A copy of the `spdata[key][data]` object with the `.X` attribute replaced by the smoothed data matrix (sparse CSR format).
	D : ndarray
		The distance/weight matrix used for smoothing.
	"""
	
	if overlap is None:
		overlap=list(range(spdata[key][X].X.shape[0]))
	else:
		overlap=spdata[key][overlap]

	
	if mode == 'Mean':
		n_neigh=50

		# Calculate spatial neighbors
		neigh=NearestNeighbors(n_neighbors=(n_neigh+1),metric='euclidean').fit(spdata[key][X][overlap].obsm['spatial'])
		dist,index=neigh.kneighbors(spdata[key][X][overlap].obsm['spatial'])
		D=np.zeros([spdata[key][X][overlap].shape[0],spdata[key][X][overlap].shape[0]])

		d=[1]*(n_neigh+1)

		for w in range(index.shape[0]):
			i=index[w][dist[w]<Dist]
			l=len(i)
			t=spdata[key][X][overlap][i]

			d=[1]*(l)
			# for z in range(1,(n_neigh+1)):
			# 	#a=math.exp(2-distance.cosine(t.obsm['X_pca'][0],t.obsm['X_pca'][z]))
			# 	d[z]=1
			d=[a/sum(d) for a in d]
			for q in range(0,l):
				D[i[0],i[q]]=d[q]
		D1=np.matmul(spdata[key][data].X.todense().T,D)

		D2=sp.csr_matrix(D1.T)
		dummy=spdata[key][data].copy()
		dummy.X=D2
	
	if mode == 'Local':

		if D is None:
			GetLocalPCADistance(spdata,key,X,Dist)

		dummy=spdata[key][data].copy()
		D=D[overlap,:][:,overlap]
		D1=0.5*dummy.X.todense().T+0.5*np.matmul(dummy.X.todense().T,D)

		D2=sp.csr_matrix(D1.T)
		dummy.X=D2
	
	if mode == 'Global':
		if D is None:
			GetGlobalPCADistance(spdata,key,X,overlap,Dist)
		
		dummy=spdata[key][data].copy()
		D=D[overlap,:][:,overlap]
		D1=0.5*dummy.X.todense().T+0.5*np.matmul(dummy.X.todense().T,D)

		D2=sp.csr_matrix(D1.T)
		dummy.X=D2
	
	return dummy, D

def rank_genes_by_correlation(expr: pd.DataFrame, scores: np.ndarray):
	"""
	Calculates the Spearman correlation between gene expression profiles and a given score vector,
	ranking genes based on the correlation coefficient.

	Args:
		expr (pd.DataFrame): A DataFrame containing gene expression data, where rows represent genes
			and columns represent samples/cells.
		scores (np.ndarray): A 1D numpy array containing scores corresponding to the samples/cells
			in the expression DataFrame.

	Returns:
		pd.Series: A Series containing the Spearman correlation coefficients for each gene,
			sorted in descending order. The index corresponds to the gene names.
	"""
	corrs = []
	for gene in expr.index:
		rho, _ = spearmanr(expr.loc[gene, :], scores)
		corrs.append(rho)
	ranking = pd.Series(corrs, index=expr.index).sort_values(ascending=False)
	return ranking