import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
import itertools
import gc
import scanpy as sc
from .utils import load_nested_keys_on_demand
from .datasets import CustomDataset, GetDataPoints
from .models import Autoencoder_Bacterial_Predictor, Autoencoder_Bacterial_Predictor_dropout, Encoder_Bacterial_Predictor_dropout, VAutoencoder_Bacterial_Predictor_dropout

import torch.nn as nn


def BacterialClassifierWorkflow(spdata_i,train_keys,Xdata,Ydata,spots,signal_cutoff,models,embed_size=128,batch_size=32,n_epochsAE=1000,n_epochs=100,kclassloss=0.1,dropout=0.2):
	"""
	Runs a comprehensive workflow for training and evaluating various machine learning models for bacterial classification.
	This function iterates through specified input data keys (`Xdata`) and target labels (`Ydata`), applies signal cutoffs, balances the dataset, and trains selected models. It supports both traditional ML models (Logistic Regression, Random Forest, XGBoost) and deep learning models (Autoencoders, Variational Autoencoders) implemented in PyTorch.
	Args:
		spdata_i (dict or str): The input spatial data dictionary or a path to a file containing the data.
		train_keys (list): List of keys identifying the training samples within `spdata_i`.
		Xdata (list of str): List of keys representing the input feature sets (e.g., 'dataX', smoothed versions).
		Ydata (list of str): List of keys representing the target label sets.
		spots (str): Key indicating the spots/locations in the data.
		signal_cutoff (list of float): List of cutoff values for signal filtering.
		models (list of str): List of model names to train. Options include:
			- 'Logistic'
			- 'RandomForest'
			- 'XGBoost'
			- 'Autoencoder_dropout'
			- 'Encoder_dropout'
		embed_size (int, optional): Size of the embedding layer for autoencoder models. Defaults to 128.
		batch_size (int, optional): Batch size for deep learning model training. Defaults to 32.
		n_epochsAE (int, optional): Number of epochs for pre-training the autoencoder component. Defaults to 1000.
		n_epochs (int, optional): Number of epochs for fine-tuning the classifier component. Defaults to 100.
		kclassloss (float, optional): Weighting factor for the classification loss in combined loss functions. Defaults to 0.1.
		dropout (float, optional): Dropout rate for models with dropout layers. Defaults to 0.2.
	Returns:
		dict: A nested dictionary containing the results for each combination of input suffix, target label, and signal cutoff.
			  Structure: OUTPUT[suffix][label][cutoff][model_name] = {'model': ..., 'auc': ..., 'accuracy': ..., etc.}
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	AE={}
	AED={}
	VAED={}
	OUTPUT={}
	print('Available Models: Logistic, RandomForest, XGBoost,  Autoencoder_dropout', 'Encoder_dropout')
	print(f'Executing Models:{models}')
	for x in Xdata:
		if x=='dataX':
			suff='Raw'
		else:
			suff=x.split('dataX', 1)[-1].lower()
		
		
		OUTPUT[suff]={}


		for y in Ydata:
			if "NoSmooth" in y:
				y=y
				print(f"Using {y} as is")
			else:
				y=y+suff
				print(f"Using {y} as {suff} smoothed")
			

			OUTPUT[suff][y]={}
			for cc in signal_cutoff:
				OUTPUT[suff][y][cc]={}
				print(f"Training for {x} and {y} at cutoff {cc}")


				if isinstance(spdata_i, str):
					print(f"Loading {spdata_i}")
					spdata=load_nested_keys_on_demand(train_keys,[x,y,spots,'xcol'],path=spdata_i)
					print(spdata.keys())
				else:
					spdata=spdata_i


				X,Y,K,Q=GetDataPoints(spdata,train_keys,Xdata=x,Ydata=y,spots=spots,signal_cutoff=cc)
				# Separate positive and negative samples
				positive_indices = [i for i, y in enumerate(Y) if y == 1]
				negative_indices = [i for i, y in enumerate(Y) if y == 0]

				# Determine the minimum count between positive and negative samples
				min_count = min(len(positive_indices), len(negative_indices))
				print(f"Positive samples: {len(positive_indices)}, Negative samples: {len(negative_indices)}, Minimum count: {min_count} - Quantile: {Q}")
				# Randomly sample from both positive and negative samples to create balanced datasets
				random.seed(42)
				balanced_positive_indices = random.sample(positive_indices, min_count)
				balanced_negative_indices = random.sample(negative_indices, min_count)

				# Combine the balanced indices
				balanced_indices = balanced_positive_indices + balanced_negative_indices
				random.shuffle(balanced_indices)

				# Create balanced dataset
				balanced_X_list = [X[i] for i in balanced_indices]
				balanced_Y_list = [Y[i] for i in balanced_indices]
				balanced_KEY = [K[i] for i in balanced_indices]

				X_train,X_test,Y_train,Y_test,K_train,K_test=train_test_split(balanced_X_list, balanced_Y_list, balanced_KEY, test_size=0.2, random_state=42)

				X_trainS=np.vstack(X_train)
				X_testS=np.vstack(X_test)
				Y_trainS=np.array(Y_train)
				Y_testS=np.array(Y_test)
				K_trainS=np.array(K_train)
				K_testS=np.array(K_test)

				if 'Logistic' in models:
					print(f"Training Logistic Regression for {suff} and {y} at cutoff {cc}")
					# Create an instance of LogisticRegression
					logreg = LogisticRegression()

					# Fit the model on the training data
					logreg.fit(X_trainS, Y_trainS)

					# Predict the labels for the test data
					Y_pred = logreg.predict_proba(X_testS)

					auc=roc_auc_score(Y_testS, Y_pred[:, 1])

					# Save the results
					OUTPUT[suff][y][cc]['Logistic'] = {'model': logreg,'auc': auc}
				
				if 'RandomForest' in models:
					print(f"Training Random Forest for {suff} and {y} at cutoff {cc}")
					# Create an instance of RandomForestClassifier
					rf = RandomForestClassifier()

					# Fit the model on the training data
					rf.fit(X_trainS, Y_trainS)

					# Predict the labels for the test data
					Y_pred = rf.predict_proba(X_testS)

					auc=roc_auc_score(Y_testS, Y_pred[:, 1])
	
					# Save the results
					OUTPUT[suff][y][cc]['RandomForest'] = {'model' : rf,'auc': auc}

				if 'XGBoost' in models:
					print(f"Training XGBoost for {suff} and {y} at cutoff {cc}")
					# Create an instance of XGBClassifier
					xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

					# Fit the model on the training data
					xgb_model.fit(X_trainS, Y_trainS)

					# Predict the labels for the test data
					Y_pred = xgb_model.predict_proba(X_testS)
					auc=roc_auc_score(Y_testS, Y_pred[:, 1])

					# Save the results
					OUTPUT[suff][y][cc]['XGBoost'] = {'model': xgb_model,'auc': auc}

				if 'Autoencoder_dropout' in models:
					print(f"Training Autoencoder with dropout for {suff} and {y} at cutoff {cc}")
					# Define the input size
					n_input = X_trainS.shape[1]

					X_train_sub,X_val_sub,Y_train_sub,Y_val_sub=train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

					# Define the dataset
					train_dataset = CustomDataset(X_train_sub, Y_train_sub)
					val_dataset = CustomDataset(X_val_sub, Y_val_sub)
					test_dataset = CustomDataset(X_test, Y_test)

					train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
					val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
					test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

					# Define the model

					if suff in AED:
						model = Autoencoder_Bacterial_Predictor_dropout(n_input, embed_size, h_dim=128,dropout=dropout)
						model = model.to(device)
						#model.ae.load_state_dict(AED[suff].ae.state_dict())
						model.ae.load_state_dict(AED[suff])
						print(f"Using pre-trained Autoencoder weights for {suff}, predictor reinitialized")
					else:
						print(f"No pre-trained Autoencoder weights found for {suff}, training from scratch")
						model = Autoencoder_Bacterial_Predictor_dropout(n_input, embed_size, h_dim=128,dropout=dropout)
						model=model.to(device)
						# Define the loss function
						criterion = nn.MSELoss()

						# Define the optimizer
						optimizer = torch.optim.Adam(model.ae.parameters(), lr=0.001)

						best_loss = float('inf')
						epochs_no_improve = 0


						# Train the model
						for epoch in range(n_epochsAE):
							model.train()
							epoch_loss = 0
							num_batches = 0
							for i, (X_batch, Y_batch) in enumerate(train_loader):
								# Move the data to the device
								X_batch = X_batch.to(device).float()
								Y_batch = Y_batch.to(device).float()

								# Zero the gradient
								optimizer.zero_grad()

								# Forward pass
								_, X = model.ae(X_batch)

								# Calculate the loss
								loss = criterion(X, X_batch)

								# Backward pass
								loss.backward()

								optimizer.step()

								epoch_loss += loss.item()
								num_batches += 1
							
							epoch_loss=epoch_loss/num_batches
							if epoch % (n_epochsAE/10) == 0:
								print(f"Epoch {epoch} Loss: {epoch_loss}")
							
							if epoch_loss < best_loss:
								best_loss = epoch_loss
								epochs_no_improve = 0
							else:
								epochs_no_improve += 1
							if epochs_no_improve == 20:
								break

						print(f"Epoch {epoch} Loss: {epoch_loss}")
						ae_d=model.ae.state_dict()



						AED[suff] = {k: v.cpu() for k, v in ae_d.items()}
						del ae_d


					criterion_classification = nn.BCELoss()
					optimizer= torch.optim.Adam(model.parameters(), lr=0.001)

					best_loss = float('inf')
					epochs_no_improve = 0

					for epoch in range(n_epochs):
						model.train()
						num_batches = 0
						num_val_batches = 0
						epoch_loss = 0
						epoch_rec_loss = 0
						epoch_class_loss = 0
						val_epoch_loss = 0
						val_epoch_rec_loss = 0
						val_epoch_class_loss = 0
						num_batches = 0
						for i, (X_batch, Y_batch) in enumerate(train_loader):
							# Move the data to the device
							X_batch = X_batch.to(device).float()
							Y_batch = Y_batch.to(device).float()

							# Zero the gradient
							optimizer.zero_grad()

							# Forward pass
							_, X, Y_pred = model(X_batch)

							Y_pred = Y_pred.squeeze()
							# Calculate the loss
							rec_loss = criterion(X, X_batch)
							class_loss = criterion_classification(Y_pred, Y_batch)
							loss = rec_loss + kclassloss*class_loss

							# Backward pass
							loss.backward()
							optimizer.step()

							num_batches += 1
						
						model.eval()
						for z, (X_val, Y_val) in enumerate(val_loader):
							X_val = X_val.to(device).float()
							Y_val = Y_val.to(device).float()
							_, X_val_hat, Y_val_pred = model(X_val)

							Y_val_pred = Y_val_pred.squeeze()

							num_val_batches += 1
							val_rec_loss = criterion(X_val_hat, X_val)
							val_class_loss = criterion_classification(Y_val_pred, Y_val)
							val_loss = val_rec_loss + kclassloss*val_class_loss
							val_epoch_loss += val_loss.item()
							val_epoch_rec_loss += val_rec_loss.item()
							val_epoch_class_loss += val_class_loss.item()
							epoch_rec_loss += rec_loss.item()
							epoch_class_loss += class_loss.item()
							epoch_loss += loss.item()

						val_epoch_loss = val_epoch_loss / num_val_batches
						val_epoch_rec_loss = val_epoch_rec_loss / num_val_batches
						val_epoch_class_loss = val_epoch_class_loss / num_val_batches
						epoch_loss = epoch_loss / num_batches
						epoch_rec_loss = epoch_rec_loss / num_batches
						epoch_class_loss = epoch_class_loss / num_batches

						if val_epoch_class_loss < best_loss:
							best_loss = val_epoch_class_loss
							epochs_no_improve = 0
						else:
							epochs_no_improve += 1
						if epochs_no_improve == 5:

							print(f"stopped at {epoch}")

							print(f"Epoch {epoch} Train Loss: {epoch_loss} Val Loss: {val_epoch_loss}")
							print(f"Epoch {epoch} Train Rec Loss: {epoch_rec_loss} Val Rec Loss: {val_epoch_rec_loss}")
							print(f"Epoch {epoch} Train Class Loss: {epoch_class_loss} Val Class Loss: {val_epoch_class_loss}")

							break
						

						if epoch % (n_epochs/10) == 0:
							print(f"Epoch {epoch} Train Loss: {epoch_loss} Val Loss: {val_epoch_loss}")
							print(f"Epoch {epoch} Train Rec Loss: {epoch_rec_loss} Val Rec Loss: {val_epoch_rec_loss}")
							print(f"Epoch {epoch} Train Class Loss: {epoch_class_loss} Val Class Loss: {val_epoch_class_loss}")

					
					# Choose cutoff to maximize F1 score

					model.eval()
					Y_valSS=[]
					Y_val_predSS=[]
					with torch.no_grad():
						for X_val, Y_val in val_loader:
							X_val = X_val.to(device).float()
							Y_val = Y_val.to(device).float()

							_, X_val_hat, Y_val_pred = model(X_val)

							# NO threshold qui
							Y_valS = Y_val.detach().cpu().numpy()
							Y_val_predS = Y_val_pred.detach().cpu().numpy().squeeze()

							Y_valSS.append(Y_valS)
							Y_val_predSS.append(Y_val_predS)

					# Concateno tutto
					Y_val_true = np.concatenate(Y_valSS)
					Y_val_probs = np.concatenate(Y_val_predSS)

					# Trovo il cutoff ottimale che massimizza F1
					thresholds = np.linspace(0, 1, 100)
					f1_scores = [f1_score(Y_val_true, Y_val_probs >= t) for t in thresholds]
					best_threshold = thresholds[np.argmax(f1_scores)]
					auc_val=roc_auc_score(Y_val_true, Y_val_probs)


					
					# Evaluate the model

					


					#model.eval()
					Y_testSS=[]
					Y_test_predSS=[]
					for t, (X_test, Y_test) in enumerate(test_loader):
						X_test = X_test.to(device).float()
						Y_test = Y_test.to(device).float()
						_, X_test_hat, Y_test_pred = model(X_test)

						#Y_test_pred = (Y_test_pred.squeeze() > 0.5).float()
						Y_test_pred = (Y_test_pred.squeeze()).float()
						Y_test_predS=Y_test_pred.detach().cpu().numpy().flatten()
						Y_testS=Y_test.detach().cpu().numpy().flatten()
						Y_testSS.extend(Y_testS)  # usa extend invece di append
						Y_test_predSS.extend(Y_test_predS)

					Y_t = Y_testSS
					Y_p = Y_test_predSS

					auc_test=roc_auc_score(Y_t, Y_p)

					
					auc_test=roc_auc_score(Y_t, Y_p)



					final_dict = model.state_dict()
					final_dict_cpu= {k: v.cpu() for k, v in final_dict.items()}
					del final_dict
					# Save the results
					OUTPUT[suff][y][cc]['Autoencoder_dropout'] = {'model': final_dict_cpu,'auc_test': auc_test, 'auc_val':auc_val}
					del model
					del optimizer
					del train_loader, val_loader, test_loader
					gc.collect()
					torch.cuda.empty_cache()
					print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
					print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

				if 'Encoder_dropout' in models:
					print(f"Training Autoencoder with dropout for {suff} and {y} at cutoff {cc}")
					# Define the input size
					n_input = X_trainS.shape[1]

					X_train_sub,X_val_sub,Y_train_sub,Y_val_sub=train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

					# Define the dataset
					train_dataset = CustomDataset(X_train_sub, Y_train_sub)
					val_dataset = CustomDataset(X_val_sub, Y_val_sub)
					test_dataset = CustomDataset(X_test, Y_test)

					train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
					val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
					test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

					# Define the model

					model = Encoder_Bacterial_Predictor_dropout(n_input, embed_size, h_dim=128,dropout=dropout)
					model=model.to(device)
					# Define the loss function

					criterion_classification = nn.BCELoss()
					optimizer= torch.optim.Adam(model.parameters(), lr=0.001)

					best_loss = float('inf')
					epochs_no_improve = 0

					for epoch in range(n_epochs):
						model.train()
						num_batches = 0
						num_val_batches = 0
						epoch_loss = 0
						epoch_rec_loss = 0
						epoch_class_loss = 0
						val_epoch_loss = 0
						val_epoch_rec_loss = 0
						val_epoch_class_loss = 0
						num_batches = 0
						for i, (X_batch, Y_batch) in enumerate(train_loader):
							# Move the data to the device
							X_batch = X_batch.to(device).float()
							Y_batch = Y_batch.to(device).float()

							# Zero the gradient
							optimizer.zero_grad()

							# Forward pass
							_, Y_pred = model(X_batch)

							Y_pred = Y_pred.squeeze()
							# Calculate the loss
							loss = criterion_classification(Y_pred, Y_batch)

							# Backward pass
							loss.backward()
							optimizer.step()

							num_batches += 1
						
						model.eval()
						for z, (X_val, Y_val) in enumerate(val_loader):
							X_val = X_val.to(device).float()
							Y_val = Y_val.to(device).float()
							_, Y_val_pred = model(X_val)

							Y_val_pred = Y_val_pred.squeeze()

							num_val_batches += 1

							val_loss = criterion_classification(Y_val_pred, Y_val)
							val_epoch_loss += val_loss.item()
							epoch_loss += loss.item()

						val_epoch_loss = val_epoch_loss / num_val_batches
						epoch_loss = epoch_loss / num_batches

						if val_epoch_loss < best_loss:
							best_loss = val_epoch_loss
							epochs_no_improve = 0
						else:
							if epoch > 30:
								epochs_no_improve += 1
						
						if epochs_no_improve == 5:

							print(f"stopped at {epoch}")

							print(f"Epoch {epoch} Train Loss: {epoch_loss} Val Loss: {val_epoch_loss}")

							break
						

						if epoch % (n_epochs/10) == 0:
							print(f"Epoch {epoch} Train Loss: {epoch_loss} Val Loss: {val_epoch_loss}")

		
					
					# Evaluate the model

					


					#model.eval()
					Y_testSS=[]
					Y_test_predSS=[]
					for t, (X_test, Y_test) in enumerate(test_loader):
						X_test = X_test.to(device).float()
						Y_test = Y_test.to(device).float()
						_, Y_test_pred = model(X_test)

						#Y_test_pred = (Y_test_pred.squeeze() > 0.5).float()
						Y_test_pred = (Y_test_pred.squeeze()).float()
						Y_test_predS=Y_test_pred.detach().cpu().numpy().flatten()
						Y_testS=Y_test.detach().cpu().numpy().flatten()
						Y_testSS.extend(Y_testS)  # usa extend invece di append
						Y_test_predSS.extend(Y_test_predS)

					Y_t = Y_testSS
					Y_p = Y_test_predSS

					auc_test=roc_auc_score(Y_t, Y_p)

					print("AUC Internal Test Encoder_dropout:", auc_test)

					final_dict = model.state_dict()
					final_dict_cpu= {k: v.cpu() for k, v in final_dict.items()}
					del final_dict
					# Save the results
					OUTPUT[suff][y][cc]['Encoder_dropout'] = {'model': final_dict_cpu,'auc_test': auc_test}
					del model
					del optimizer
					del train_loader, val_loader, test_loader
					gc.collect()
					torch.cuda.empty_cache()
					print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
					print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

	return OUTPUT