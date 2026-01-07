import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
	"""
	A simple feed-forward neural network encoder.
	This module compresses an input vector into a lower-dimensional embedding space
	through a series of linear layers and ReLU activations.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The size of the output embedding vector (latent space).
	Attributes:
		enc_1 (nn.Linear): First linear layer transforming input to 1024 dimensions.
		enc_2 (nn.Linear): Second linear layer transforming 1024 to 512 dimensions.
		enc_3 (nn.Linear): Third linear layer transforming 512 to 256 dimensions.
		z_layer (nn.Linear): Final linear layer transforming 256 to `embed_size`.
	Methods:
		forward(x): Performs the forward pass of the encoder.
	"""
	def __init__(self,n_input, embed_size):
		super(Encoder, self).__init__()
		self.enc_1 = (nn.Linear(n_input, 1024))
		self.enc_2 = (nn.Linear(1024, 512))
		self.enc_3 = (nn.Linear(512, 256))
		self.z_layer = (nn.Linear(256, embed_size))

	def forward(self, x):
		x = F.relu(self.enc_1(x))
		x = F.relu(self.enc_2(x))
		x = F.relu(self.enc_3(x))
		x = self.z_layer(x)
		return x

class Encoder_dropout(nn.Module):
	"""
	A neural network encoder module with dropout regularization.
	This class implements a multi-layer perceptron (MLP) encoder that compresses an input vector
	into a lower-dimensional embedding space. It consists of three hidden layers with ReLU activations
	and dropout applied after each hidden layer, followed by a final linear projection layer.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The size of the output embedding vector (latent space dimension).
		dropout (float, optional): The dropout probability applied after each hidden layer. Defaults to 0.2.
	Attributes:
		enc_1 (nn.Linear): First linear layer mapping input to 1024 units.
		enc_2 (nn.Linear): Second linear layer mapping 1024 units to 512 units.
		enc_3 (nn.Linear): Third linear layer mapping 512 units to 256 units.
		z_layer (nn.Linear): Final linear layer mapping 256 units to `embed_size`.
		dropout (nn.Dropout): Dropout layer used for regularization.
	Methods:
		forward(x): Defines the forward pass of the encoder.
	"""
	def __init__(self,n_input, embed_size,dropout=0.2):
		super(Encoder_dropout, self).__init__()
		self.enc_1 = (nn.Linear(n_input, 1024))
		self.enc_2 = (nn.Linear(1024, 512))
		self.enc_3 = (nn.Linear(512, 256))
		self.z_layer = (nn.Linear(256, embed_size))
		self.dropout=nn.Dropout(dropout)

	def forward(self, x):
		x = F.relu(self.enc_1(x))
		x = self.dropout(x)
		x = F.relu(self.enc_2(x))
		x = self.dropout(x)
		x = F.relu(self.enc_3(x))
		x = self.dropout(x)
		x = self.z_layer(x)
		return x

class VEncoder_dropout(nn.Module):
	"""
	A Variational Autoencoder (VAE) Encoder module with dropout regularization.
	This class implements the encoder portion of a VAE, transforming an input vector
	into a latent space representation defined by a mean (mu) and a log-variance (logvar).
	It utilizes a multi-layer perceptron (MLP) architecture with ReLU activations and
	dropout layers between transformations to prevent overfitting.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The dimensionality of the latent space embedding.
		dropout (float, optional): The dropout probability applied after each hidden layer. 
								   Defaults to 0.2.
	Attributes:
		enc_1 (nn.Linear): First linear transformation layer (n_input -> 1024).
		enc_2 (nn.Linear): Second linear transformation layer (1024 -> 512).
		enc_3 (nn.Linear): Third linear transformation layer (512 -> 256).
		z1_layer (nn.Linear): Output layer for the latent mean vector (mu) (256 -> embed_size).
		z2_layer (nn.Linear): Output layer for the latent log-variance vector (logvar) (256 -> embed_size).
		dropout (nn.Dropout): Dropout layer applied for regularization.
	Methods:
		forward(x):
			Performs the forward pass of the encoder.
			Args:
				x (torch.Tensor): Input tensor of shape (batch_size, n_input).
			Returns:
				tuple: A tuple containing:
					- mu (torch.Tensor): The mean of the latent distribution (batch_size, embed_size).
					- logvar (torch.Tensor): The log-variance of the latent distribution (batch_size, embed_size).
	"""
	def __init__(self,n_input, embed_size,dropout=0.2):
		super(VEncoder_dropout, self).__init__()
		self.enc_1 = (nn.Linear(n_input, 1024))
		self.enc_2 = (nn.Linear(1024, 512))
		self.enc_3 = (nn.Linear(512, 256))
		self.z1_layer = (nn.Linear(256, embed_size))
		self.z2_layer = (nn.Linear(256, embed_size))
		self.dropout=nn.Dropout(dropout)

	def forward(self, x):
		x = F.relu(self.enc_1(x))
		x = self.dropout(x)
		x = F.relu(self.enc_2(x))
		x = self.dropout(x)
		x = F.relu(self.enc_3(x))
		x = self.dropout(x)
		mu = self.z1_layer(x)
		logvar = self.z2_layer(x)
		return mu, logvar

class Decoder(nn.Module):
	"""
	A simple fully connected decoder network.
	This module takes an embedding vector and upscales it through a series of linear layers
	with ReLU activations to produce an output vector of a specified size.
	Args:
		embed_size (int): The size of the input embedding vector.
		n_output (int): The size of the final output vector.
	Attributes:
		dec_1 (nn.Linear): First linear layer mapping embed_size to 256.
		dec_2 (nn.Linear): Second linear layer mapping 256 to 512.
		dec_3 (nn.Linear): Third linear layer mapping 512 to 1024.
		output_layer (nn.Linear): Final linear layer mapping 1024 to n_output.
	"""
	def __init__(self, embed_size, n_output):
		super(Decoder, self).__init__()
		self.dec_1 = (nn.Linear(embed_size, 256))
		self.dec_2 = (nn.Linear(256, 512))
		self.dec_3 = (nn.Linear(512, 1024))
		self.output_layer = (nn.Linear(1024, n_output))

	def forward(self, x):
		x = F.relu(self.dec_1(x))
		x = F.relu(self.dec_2(x))
		x = F.relu(self.dec_3(x))
		x = self.output_layer(x)
		return x

class Decoder_dropout(nn.Module):
	"""
	A neural network decoder module with dropout regularization.
	This class implements a multi-layer perceptron (MLP) decoder that progressively
	upsamples the input embedding to a target output size. It includes ReLU activation
	functions and dropout layers between linear transformations to prevent overfitting.
	Args:
		embed_size (int): The size of the input embedding vector.
		n_output (int): The size of the final output vector.
		dropout (float, optional): The probability of an element to be zeroed in the dropout layers. 
								   Defaults to 0.2.
	Attributes:
		dec_1 (nn.Linear): First linear layer transforming `embed_size` to 256.
		dec_2 (nn.Linear): Second linear layer transforming 256 to 512.
		dec_3 (nn.Linear): Third linear layer transforming 512 to 1024.
		output_layer (nn.Linear): Final linear layer transforming 1024 to `n_output`.
		dropout (nn.Dropout): Dropout layer applied after activations.
	Methods:
		forward(x): Defines the forward pass of the decoder.
	"""
	def __init__(self, embed_size, n_output,dropout=0.2):
		super(Decoder_dropout, self).__init__()
		self.dec_1 = (nn.Linear(embed_size, 256))
		self.dec_2 = (nn.Linear(256, 512))
		self.dec_3 = (nn.Linear(512, 1024))
		self.output_layer = (nn.Linear(1024, n_output))
		self.dropout=nn.Dropout(dropout)

	def forward(self, x):
		x = F.relu(self.dec_1(x))
		x = self.dropout(x)
		x = F.relu(self.dec_2(x))
		x = self.dropout(x)
		x = F.relu(self.dec_3(x))
		x = self.dropout(x)
		x = self.output_layer(x)
		return x


class Autoencoder(nn.Module):
	"""
	A simple Autoencoder neural network architecture implemented in PyTorch.
	This class combines an Encoder and a Decoder module to learn a compressed 
	representation (embedding) of the input data and reconstruct it.
	Args:
		n_input (int): The dimensionality of the input features.
		embed_size (int): The dimensionality of the latent space (embedding).
	Attributes:
		encoder (Encoder): The encoder module that maps input to the latent space.
		decoder (Decoder): The decoder module that reconstructs input from the latent space.
	Methods:
		forward(x): Performs a forward pass through the network.
	"""
	def __init__(self, n_input, embed_size):
		super(Autoencoder, self).__init__()
		self.encoder = Encoder(n_input, embed_size)
		self.decoder = Decoder(embed_size, n_input)

	def forward(self, x):
		z = self.encoder(x)
		x = self.decoder(z)
		return z,x


class Autoencoder_dropout(nn.Module):
	"""
	A PyTorch module implementing an Autoencoder with dropout layers.
	This class combines an encoder and a decoder, both incorporating dropout for regularization,
	to learn a compressed representation of the input data.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The size of the latent space (embedding) representation.
		dropout (float, optional): The dropout probability. Defaults to 0.2.
	Attributes:
		encoder (Encoder_dropout): The encoder module that maps input to latent space.
		decoder (Decoder_dropout): The decoder module that reconstructs input from latent space.
	Methods:
		forward(x): Performs a forward pass through the autoencoder.
	Returns:
		tuple: A tuple containing:
			- z (torch.Tensor): The latent representation (embedding) of the input.
			- x (torch.Tensor): The reconstructed input.
	"""
	def __init__(self, n_input, embed_size,dropout=0.2):
		super(Autoencoder_dropout, self).__init__()
		self.encoder = Encoder_dropout(n_input, embed_size,dropout)
		self.decoder = Decoder_dropout(embed_size, n_input,dropout)

	def forward(self, x):
		z = self.encoder(x)
		x = self.decoder(z)
		return z,x


class VAutoencoder_dropout(nn.Module):
	"""
	A Variational Autoencoder (VAE) with dropout regularization.
	This module implements a standard VAE architecture consisting of an encoder and a decoder,
	incorporating dropout layers for regularization. It learns a latent representation of the
	input data by optimizing the Evidence Lower Bound (ELBO).
	Args:
		n_input (int): The dimensionality of the input features.
		embed_size (int): The dimensionality of the latent space (embedding size).
		dropout (float, optional): The dropout probability used in both the encoder and decoder.
			Defaults to 0.2.
	Attributes:
		encoder (VEncoder_dropout): The encoder module that maps inputs to latent distribution parameters (mu, logvar).
		decoder (Decoder_dropout): The decoder module that reconstructs inputs from latent vectors.
	Methods:
		reparametrize(mu, logvar):
			Performs the reparameterization trick to sample from the latent distribution
			while allowing backpropagation. It clamps log-variance for numerical stability.
		forward(x):
			Performs a forward pass through the network.
			1. Encodes input `x` to `mu` and `logvar`.
			2. Samples latent vector `z` using reparameterization.
			3. Decodes `z` to reconstruct `x`.
	Returns:
		tuple: A tuple containing:
			- mu (torch.Tensor): Mean of the latent Gaussian distribution.
			- logvar (torch.Tensor): Log-variance of the latent Gaussian distribution.
			- x (torch.Tensor): The reconstructed output.
	"""
	def __init__(self, n_input, embed_size,dropout=0.2):
		super(VAutoencoder_dropout, self).__init__()
		self.encoder = VEncoder_dropout(n_input, embed_size,dropout)
		self.decoder = Decoder_dropout(embed_size, n_input,dropout)

	def reparametrize(self, mu, logvar):
		logvar = torch.clamp(logvar, min=-10, max=10) 
		std= torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparametrize(mu, logvar)
		x_recon = self.decoder(z)
		return mu, logvar, z, x_recon



class Encoder_Bacterial_Predictor_dropout(nn.Module):
	"""
	A neural network module combining an encoder with a bacterial predictor head, incorporating dropout.
	This model consists of two main components:
	1. An encoder (`Encoder_dropout`) that transforms the input into a latent embedding.
	2. A predictor (`nn.Sequential`) that takes the embedding and predicts a probability score (0-1) using a hidden layer and a sigmoid activation.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The size of the output embedding from the encoder.
		h_dim (int): The size of the hidden layer in the predictor network.
		dropout (float, optional): The dropout probability used in both the encoder and the predictor. Defaults to 0.2.
	Attributes:
		encoder (Encoder_dropout): The encoder submodule.
		predictor (nn.Sequential): The predictor submodule consisting of Linear -> ReLU -> Dropout -> Linear -> Sigmoid.
	Methods:
		forward(x):
			Passes the input through the encoder and then the predictor.
			Args:
				x (torch.Tensor): Input tensor of shape (batch_size, n_input).
			Returns:
				tuple: A tuple containing:
					- z (torch.Tensor): The latent embedding of shape (batch_size, embed_size).
					- pred (torch.Tensor): The prediction score of shape (batch_size, 1).
	"""
	def __init__(self, n_input, embed_size, h_dim,dropout=0.2):
		super(Encoder_Bacterial_Predictor_dropout, self).__init__()
		self.encoder = Encoder_dropout(n_input, embed_size,dropout)
		self.predictor = nn.Sequential(
			nn.Linear(embed_size, h_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(h_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		z = self.encoder(x)
		pred = self.predictor(z)
		return z, pred




class Autoencoder_Bacterial_Predictor(nn.Module):
	"""
	A neural network module that combines an Autoencoder with a bacterial prediction head.
	This model utilizes an autoencoder to learn a compressed representation (embedding)
	of the input data and simultaneously predicts a probability score (e.g., bacterial presence)
	based on that embedding.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The size of the latent space (embedding) produced by the autoencoder.
		h_dim (int): The size of the hidden layer in the predictor network.
	Attributes:
		ae (Autoencoder): The underlying autoencoder instance used for feature extraction and reconstruction.
		predictor (nn.Sequential): A feed-forward neural network that maps the embedding to a scalar probability.
	Forward Args:
		x (torch.Tensor): Input tensor of shape (batch_size, n_input).
	Returns:
		tuple: A tuple containing:
			- z (torch.Tensor): The latent embedding of shape (batch_size, embed_size).
			- x_recon (torch.Tensor): The reconstructed input of shape (batch_size, n_input).
			- pred (torch.Tensor): The prediction score between 0 and 1 of shape (batch_size, 1).
	"""
	def __init__(self, n_input, embed_size, h_dim):
		super(Autoencoder_Bacterial_Predictor, self).__init__()
		self.ae=Autoencoder(n_input, embed_size)
		self.predictor = nn.Sequential(
			nn.Linear(embed_size, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		z,x = self.ae(x)
		pred = self.predictor(z)
		return z,x,pred

class Autoencoder_Bacterial_Predictor_dropout(nn.Module):
	"""
	A neural network module combining an autoencoder with a bacterial prediction head, incorporating dropout for regularization.
	This model processes input data through an autoencoder to generate a latent representation (embedding),
	which is then reconstructed and simultaneously fed into a prediction network to output a probability score.
	Args:
		n_input (int): The size of the input feature vector.
		embed_size (int): The size of the latent space (embedding) produced by the autoencoder.
		h_dim (int): The size of the hidden layer in the predictor network.
		dropout (float, optional): The dropout probability used in both the autoencoder and the predictor. Defaults to 0.2.
	Attributes:
		ae (Autoencoder_dropout): The underlying autoencoder module responsible for encoding and decoding the input.
		predictor (nn.Sequential): A feed-forward network that predicts a score based on the latent embedding.
								   It consists of a Linear layer, ReLU activation, Dropout, a second Linear layer, and Sigmoid activation.
	Forward Args:
		x (torch.Tensor): The input tensor of shape (batch_size, n_input).
	Returns:
		tuple: A tuple containing:
			- z (torch.Tensor): The latent embedding of shape (batch_size, embed_size).
			- x (torch.Tensor): The reconstructed input of shape (batch_size, n_input).
			- pred (torch.Tensor): The prediction score of shape (batch_size, 1), bounded between 0 and 1.
	"""
	def __init__(self, n_input, embed_size, h_dim,dropout=0.2):
		super(Autoencoder_Bacterial_Predictor_dropout, self).__init__()
		self.ae=Autoencoder_dropout(n_input, embed_size,dropout)
		self.predictor = nn.Sequential(
			nn.Linear(embed_size, h_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(h_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		z,x = self.ae(x)
		pred = self.predictor(z)
		return z,x,pred




class Autoencoder_Bacterial_Regressor_dropout(nn.Module):
	"""
	A neural network module combining an autoencoder with a regression predictor, incorporating dropout.
	This class integrates a pre-defined `Autoencoder_dropout` model to extract latent features (embeddings)
	from the input data, which are then fed into a sequential regression head. It is designed to simultaneously
	reconstruct the input and predict a target variable, useful for multi-task learning scenarios involving
	bacterial data.
	Args:
		n_input (int): The number of input features.
		n_output (int): The number of output units for the regression predictor.
		embed_size (int): The size of the latent embedding vector produced by the autoencoder.
		h_dim (int): The size of the hidden layer in the predictor network.
		dropout (float, optional): The dropout probability used in both the autoencoder and the predictor.
			Defaults to 0.2.
	Attributes:
		ae (Autoencoder_dropout): The underlying autoencoder instance.
		predictor (nn.Sequential): A feed-forward neural network that maps embeddings to the output prediction.
	Forward Args:
		x (torch.Tensor): Input tensor of shape (batch_size, n_input).
	Returns:
		tuple: A tuple containing:
			- z (torch.Tensor): The latent embedding of shape (batch_size, embed_size).
			- x (torch.Tensor): The reconstructed input of shape (batch_size, n_input).
			- pred (torch.Tensor): The regression prediction of shape (batch_size, n_output).
	"""
	def __init__(self, n_input,n_output, embed_size, h_dim,dropout=0.2):
		super(Autoencoder_Bacterial_Regressor_dropout, self).__init__()
		self.ae=Autoencoder_dropout(n_input, embed_size,dropout)
		self.predictor = nn.Sequential(
			nn.Linear(embed_size, h_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(h_dim, n_output),
		)

	def forward(self, x):
		z,x = self.ae(x)
		pred = self.predictor(z)
		return z,x,pred



class VAutoencoder_Bacterial_Predictor_dropout(nn.Module):
	"""
	A Variational Autoencoder (VAE) combined with a bacterial predictor network with dropout regularization.
	This model integrates a VAE for feature extraction/dimensionality reduction with a feed-forward
	neural network for binary classification (prediction). It is designed to reconstruct the input
	while simultaneously predicting a target value based on the latent representation.
	Args:
		n_input (int): The dimensionality of the input data.
		embed_size (int): The dimensionality of the latent space (z).
		h_dim (int): The size of the hidden layer in the predictor network.
		dropout (float, optional): The dropout probability used in both the VAE and the predictor.
								   Defaults to 0.2.
	Attributes:
		vae (VAutoencoder_dropout): The underlying Variational Autoencoder module.
		predictor (nn.Sequential): A sequential network that maps the latent embedding to a
								   probability score (0-1).
	Forward Args:
		x (torch.Tensor): Input tensor of shape (batch_size, n_input).
	Returns:
		tuple: A tuple containing:
			- mu (torch.Tensor): The mean of the latent distribution.
			- logvar (torch.Tensor): The log variance of the latent distribution.
			- z (torch.Tensor): The sampled latent vector using the reparameterization trick.
			- x (torch.Tensor): The reconstructed input from the VAE.
			- pred (torch.Tensor): The prediction score from the predictor network (sigmoid output).
	"""
	def __init__(self, n_input, embed_size, h_dim,dropout=0.2):
		super(VAutoencoder_Bacterial_Predictor_dropout, self).__init__()
		self.vae=VAutoencoder_dropout(n_input, embed_size,dropout)
		self.predictor = nn.Sequential(
			nn.Linear(embed_size, h_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(h_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		mu, logvar, z, x_recon = self.vae(x)
		pred = self.predictor(z)
		return  mu, logvar, z, x_recon, pred

