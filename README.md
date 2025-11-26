# PatHoST

**Pat**hogen-**Ho**st **S**patial **T**ranscriptomics

PatHoST is a Python framework for spatial transcriptomics data analysis, designed to study pathogen-host interactions using machine learning and deep learning techniques.

## 📋 Description

PatHoST provides tools for:

- **Bacterial Classification**: Machine learning and deep learning models for classifying pathogen presence/absence in spatial data
- **Spatial Smoothing**: Algorithms for data smoothing based on spatial proximity and PCA similarity
- **Ground Truth Projection**: Alignment and projection of ground truth data onto training datasets
- **Performance Evaluation**: Clustering metrics and model stability assessment

## 🏗️ Model Architecture

The library includes various neural network architectures:

| Model | Description |
|-------|-------------|
| `Encoder` / `Encoder_dropout` | Feed-forward encoder for latent space compression |
| `Decoder` / `Decoder_dropout` | Decoder for data reconstruction |
| `Autoencoder` / `Autoencoder_dropout` | Standard autoencoder with/without dropout |
| `Autoencoder_Bacterial_Predictor` | Autoencoder with classification head for bacterial prediction |
| `Encoder_Bacterial_Predictor_dropout` | Encoder with integrated predictor |

## 📁 Project Structure

```
PatHoST/
├── src/
│   ├── __init__.py
│   ├── models.py        # Neural network architecture definitions
│   ├── train.py         # Training workflow for classification
│   ├── predict.py       # Prediction functions
│   ├── datasets.py      # Dataset and data loading utilities
│   ├── spatial.py       # Spatial analysis functions
│   ├── metrics.py       # Evaluation metrics and clustering
│   └── utils.py         # General utilities
├── notebooks/
│   ├── Data_prep.ipynb  # Data preparation
│   ├── Training.ipynb   # Model training
│   └── Evaluation.ipynb # Results evaluation
└── requirements.txt     # Dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Setup with Conda

```bash
# Create a new conda environment
conda create --name pathost python=3.13
conda activate pathost

# Install dependencies
pip install torch torchvision scanpy anndata scikit-learn xgboost
```

### Setup with pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install main dependencies
pip install torch torchvision
pip install scanpy anndata
pip install scikit-learn xgboost
pip install numpy pandas scipy matplotlib
```

## 📖 Usage


### Spatial Smoothing

```python
from src.spatial import Smooth

# Apply Mean-based smoothing
smoothed_data, distance_matrix = Smooth(
    spdata=spatial_data,
    key='sample',
    data='dataY',
    X='dataX',
    mode='Mean',
    Dist=200
)

# Or apply Local PCA-based smoothing
smoothed_data, D = Smooth(
    spdata=spatial_data,
    key='sample',
    data='dataY',
    X='dataX',
    mode='Local',
    Dist=200
)
```

### Training the Bacterial Classifier

```python
from src.train import BacterialClassifierWorkflow

# Define parameters
train_keys = ['sample_1', 'sample_2']
Xdata = ['dataX']
Ydata = ['bacterial_signal']
models = ['Logistic', 'RandomForest', 'XGBoost', 'Autoencoder_dropout']

# Run training workflow
results = BacterialClassifierWorkflow(
    spdata_i=spatial_data,
    train_keys=train_keys,
    Xdata=Xdata,
    Ydata=Ydata,
    spots='overlapping_spots',
    signal_cutoff=[0.5],
    models=models,
    embed_size=128,
    batch_size=32,
    n_epochsAE=1000,
    n_epochs=100
)
```

### Prediction

```python
from src.predict import BacterialClassifierPrediction
from src.models import Autoencoder_Bacterial_Predictor_dropout

# Load model
n_genes = spatial_data['sample']['dataX'].shape[1]  # Number of genes/features in expression matrix
model = Autoencoder_Bacterial_Predictor_dropout(n_input=n_genes, embed_size=128, h_dim=128)
model.load_state_dict(trained_weights)

# Run prediction
predictions = BacterialClassifierPrediction(
    spdata=spatial_data,
    key='test_sample',
    x='dataX',
    model=model,
    type='Autoencoder_dropout'
)
```

### Evaluation with Clustering Metrics

```python
from src.metrics import evaluate_all_methods

# Evaluate all clustering methods
results_df, labels_df = evaluate_all_methods(
    pred_scores=prediction_scores,
    true_labels=ground_truth,  # optional
    fcmin=0.3,
    fcmax=0.7
)

print(results_df)
```

## 📊 Supported Models

### Classical Machine Learning
- **Logistic Regression**: Linear classification
- **Random Forest**: Decision tree ensemble
- **XGBoost**: Optimized gradient boosting

### Deep Learning
- **Autoencoder with Dropout**: Latent representation learning with regularization
- **Encoder with Predictor**: Direct classification from latent space

## 🔧 Main Dependencies

| Package | Usage |
|---------|-------|
| `torch` | Deep learning framework |
| `scanpy` | Single-cell analysis |
| `anndata` | Transcriptomics data structure |
| `scikit-learn` | Classical machine learning |
| `xgboost` | Gradient boosting |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |

## 📓 Notebooks

The `notebooks/` folder contains interactive tutorials:

1. **Data_prep.ipynb**: Spatial data preparation and preprocessing
2. **Training.ipynb**: Model training for classification
3. **Evaluation.ipynb**: Performance evaluation and results visualization

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

*PatHoST - Advanced pathogen-host interaction analysis through spatial transcriptomics*
