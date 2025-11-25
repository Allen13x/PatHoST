import numpy as np
import pandas as pd
import sklearn.cluster as skc
import sklearn.metrics as skm
import skfuzzy as fuzz

# -------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------

def reorder_labels(scores, labels):
	"""
	Reorders labels so that label 0 corresponds to the cluster 
	with the lowest mean score, and so on.
	"""
	unique_labels = np.unique(labels)
	# Calculate mean for each cluster (handles empty cases with -inf)
	means = [scores[labels == i].mean() if np.any(labels == i) else -np.inf 
			 for i in unique_labels]
	
	# Find the order
	order = np.argsort(means)
	mapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels[order])}
	
	# Apply mapping
	return np.array([mapping[l] for l in labels])

def calculate_metrics(scores, labels, true_labels=None):
	"""
	Calculates Silhouette, Calinski-Harabasz, Davies-Bouldin.
	If true_labels is provided, calculates F1 score considering the highest cluster as positive.
	"""
	if len(np.unique(labels)) < 2:
		return {
			'silhouette': np.nan, 'calinski_harabasz': np.nan, 
			'davies_bouldin': np.nan, 'f1_score': np.nan
		}
	
	metrics = {
		'silhouette': skm.silhouette_score(scores.reshape(-1, 1), labels),
		'calinski_harabasz': skm.calinski_harabasz_score(scores.reshape(-1, 1), labels),
		'davies_bouldin': skm.davies_bouldin_score(scores.reshape(-1, 1), labels),
	}

	if true_labels is not None:
		# Assume the cluster with the highest index is the positive class (e.g. "High")
		highest_label = np.max(labels)
		binary_pred = (labels == highest_label).astype(int)
		# Ensure true_labels is binary or compatible
		metrics['f1_score'] = skm.f1_score(true_labels, binary_pred)
	else:
		metrics['f1_score'] = np.nan
		
	return metrics

# -------------------------------------------------------------------------
# CLUSTERING WRAPPERS
# -------------------------------------------------------------------------

def run_kmeans(scores, n_clusters=3, random_state=0):
	kmeans = skc.KMeans(n_clusters=n_clusters, random_state=random_state)
	labels = kmeans.fit_predict(scores.reshape(-1, 1))
	return reorder_labels(scores, labels)

def run_hierarchical(scores, n_clusters=3, linkage='ward'):
	model = skc.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
	labels = model.fit_predict(scores.reshape(-1, 1))
	return reorder_labels(scores, labels)

def run_basic_cutoff(scores, quantile=0.5):
	labels = (scores >= np.quantile(scores, quantile)).astype(int)
	return reorder_labels(scores, labels)

def run_fcm(scores, c=2, m=2.0, fcmin=0.3, fcmax=0.7, error=1e-6, maxiter=1000):
	"""
	Runs Fuzzy C-Means with 3-class logic derived from 2 centers:
	Low (0), Medium (1), High (2) based on membership thresholds.
	"""
	# FCM requires shape (n_features, n_samples)
	cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
		scores.reshape(1, -1), c=c, m=m, error=error, maxiter=maxiter
	)
	
	# Identify the "High" cluster (largest centroid)
	high_cluster_idx = np.argmax(cntr.ravel())
	membership_high = u[high_cluster_idx]
	
	# Hard assignment with threshold
	labels = np.full_like(scores, 1, dtype=int) # Default Medium
	labels[membership_high >= fcmax] = 2       # High
	labels[membership_high < fcmin] = 0        # Low
	
	return labels.astype(str) # Returns string for consistency with original code, or int if preferred

# -------------------------------------------------------------------------
# STABILITY
# -------------------------------------------------------------------------

def bootstrap_stability(scores, cluster_func, n_bootstrap=50, frac=0.8, random_state=0):
	"""
	Calculates stability (Adjusted Rand Index) via bootstrap.
	cluster_func must accept 'scores' as the only positional argument.
	"""
	np.random.seed(random_state)
	ari_scores = []
	n_samples = len(scores)
	
	for _ in range(n_bootstrap):
		# Sampling with replacement
		idx = np.random.choice(n_samples, size=int(n_samples * frac), replace=True)
		scores_bs = scores[idx]
		
		# Original labels (subset)
		# Note: we recalculate "ground truth" labels for this subset using the function
		# or, ideally, we should compare with the model fitted on the full dataset.
		# Here we replicate the logic of your script: ARI between run on bootstrap and recalculated labels.
		
		# To calculate ARI a reference is needed. In your script you compared:
		# Labels obtained on the subset vs Labels obtained by fitting AGAIN on the subset. 
		# (Algorithm self-consistency).
		
		# In your original script: adjusted_rand_score(labels_original[idx], labels_bs)
		# So labels_original is needed from outside or calculated here.
		# We modify the signature to accept labels_original if we want to do as the script,
		# but here we calculate labels_bs internally.
		
		# Correct approach for "Stability": 
		# 1. Clustering on full dataset -> labels_full
		# 2. Clustering on bootstrap -> labels_bs
		# 3. ARI between labels_full[idx] and labels_bs
		
		labels_bs = cluster_func(scores_bs)
		
		# Here is a trick: cluster_func is stateless (e.g. KMeans fit_predict every time).
		# To have labels_original we must have calculated them outside.
		# For simplicity, we recalculate labels_original inside the master function and pass them.
		pass 

	return np.nan # Placeholder, see implementation below wrapper

def run_stability_check(scores, labels_original, cluster_func_wrapper, n_bootstrap=50, frac=0.8, random_state=0):
	np.random.seed(random_state)
	ari_scores = []
	n_samples = len(scores)
	
	for i in range(n_bootstrap):
		idx = np.random.choice(n_samples, size=int(n_samples*frac), replace=True)
		scores_bs = scores[idx]
		
		# Clustering on the bootstrap sample
		labels_bs = cluster_func_wrapper(scores_bs)
		
		# ARI between original labels (of those points) and new ones
		ari = skm.adjusted_rand_score(labels_original[idx], labels_bs)
		ari_scores.append(ari)
		
	return np.mean(ari_scores), np.std(ari_scores)

# -------------------------------------------------------------------------
# MASTER FUNCTION
# -------------------------------------------------------------------------

def evaluate_all_methods(pred_scores, true_labels=None, fcmin=0.3, fcmax=0.7):
	"""
	Runs all specified clustering methods, calculates metrics and stability.
	Returns:
	1. results_df: DataFrame with metrics per method.
	2. labels_df: DataFrame with labels for each sample.
	"""
	
	# 1. Definition of methods and their execution functions
	# Use lambda to fix parameters
	methods = {
		'Median Cutoff': lambda x: run_basic_cutoff(x, 0.5),
		'KMeans 3': lambda x: run_kmeans(x, 3),
		'KMeans 2': lambda x: run_kmeans(x, 2),
		'Hierarchical 3': lambda x: run_hierarchical(x, 3),
		'Hierarchical 2': lambda x: run_hierarchical(x, 2),
		'FCM 3': lambda x: run_fcm(x, c=2, fcmin=fcmin, fcmax=fcmax) # c=2 generates 3 logical classes here
	}
	
	metrics_list = []
	all_labels = {}

	# 2. Loop over methods
	for name, func in methods.items():
		# A. Calculate Labels
		labels = func(pred_scores)
		
		# Convert to int for metrics (if strings)
		if labels.dtype.kind in {'U', 'S', 'O'}: # if string
			# map to int for scikit-learn metrics
			_, labels_int = np.unique(labels, return_inverse=True)
		else:
			labels_int = labels.astype(int)

		# Save labels for output
		all_labels[name] = labels 
		
		# B. Calculate Standard Metrics
		m = calculate_metrics(pred_scores, labels_int, true_labels)
		
		# C. Calculate Stability (Bootstrap)
		ari_mean, ari_std = run_stability_check(
			pred_scores, labels_int, 
			cluster_func_wrapper=lambda x: func(x).astype(int) if func(x).dtype.kind not in 'i' else func(x),
			n_bootstrap=50
		)
		
		# Aggregation
		row = {
			'Method': name,
			'Silhouette': m['silhouette'],
			'Calinski_Harabasz': m['calinski_harabasz'],
			'F1_Score': m['f1_score'],
			'ARI_Mean': ari_mean,
			'ARI_Std': ari_std
		}
		metrics_list.append(row)

	results_df = pd.DataFrame(metrics_list).set_index('Method')
	labels_df = pd.DataFrame(all_labels, index=range(len(pred_scores)))
	
	return results_df, labels_df