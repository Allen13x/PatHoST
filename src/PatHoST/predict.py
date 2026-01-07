import numpy as np
import torch
import scanpy as sc



def BacterialClassifierPrediction(spdata,key,x,model,type):
	"""
	Predicts bacterial classification probabilities using a specified model type.
	This function extracts feature data from a sparse matrix structure, processes it,
	and generates predictions using either Scikit-learn style models (Logistic Regression,
	Random Forest, XGBoost) or PyTorch-based neural networks (Autoencoder, Encoder).
	Args:
		spdata (dict): A dictionary containing sparse data structures. It is expected to
			have keys corresponding to `key` and nested structure accessible via `x`.
			Specifically, `spdata[key]['xcol']` is used for column selection.
		key (str): The key to access the specific dataset within `spdata`.
		x (int or str): The index or key to access the specific sample/batch within `spdata[key]`.
			The object at `spdata[key][x]` must have an `.X` attribute (sparse matrix).
		model (object): The trained machine learning or deep learning model. Can be a
			Scikit-learn estimator, XGBoost model, or a PyTorch `nn.Module`.
		type (str): The type of model being used. Supported values are:
			- 'Logistic'
			- 'RandomForest'
			- 'XGBoost'
			- 'Autoencoder_dropout'
			- 'Encoder_dropout'
	Returns:
		numpy.ndarray: An array of prediction probabilities.
			- For Scikit-learn/XGBoost models, returns the probability of the positive class (class 1).
			- For PyTorch models, returns the squeezed output tensor converted to a numpy array.
	"""


	X_list=[]
	#print(f"Extracting from {key}")

	q=spdata[key][x].X.toarray()
	q=q[:,spdata[key]['xcol']]
	X_list.append(q)

	XS=np.vstack(X_list)

	if type=='Logistic':
		Y_pred = model.predict_proba(XS)
		predictions = Y_pred
		predictions = predictions[:, 1]
	
	if type=='RandomForest':
		Y_pred = model.predict_proba(XS)
		predictions = Y_pred
		predictions = predictions[:, 1]
	
	if type=='XGBoost':
		Y_pred = model.predict_proba(XS)
		predictions = Y_pred
		predictions = predictions[:, 1]
	
	if type=='Autoencoder_dropout':
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model=model.to(device)
		model.eval()
		XS=torch.tensor(XS, dtype=torch.float32)
		XS=XS.to(device).float()
		_, _, Y_pred = model(XS)

		predictions = Y_pred.squeeze().detach().cpu().numpy()
	
	if type=='Encoder_dropout':
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model=model.to(device)
		model.eval()
		XS=torch.tensor(XS, dtype=torch.float32)
		XS=XS.to(device).float()
		_, Y_pred = model(XS)

		predictions = Y_pred.squeeze().detach().cpu().numpy()
	
	return predictions
