# Conformer Models for EEG-Based Memory Classification

This directory contains implementations of Conformer-based neural network architectures for classifying memory retention in EEG data from second language word learning experiments.

## Directory Structure

### conformer/
Standard Conformer implementation with time-domain EEG processing.

**Files:**
- `conformer.py`: Base Conformer model architecture with patch embedding and transformer encoder
- `EEG-conformer_wopreprocessing_smote.ipynb`: Training pipeline with SMOTE oversampling
- `EEG-conformer_wopreprocessing_undersampler.ipynb`: Training pipeline with random undersampling

### conformer_optimized/
Enhanced Conformer implementation incorporating frequency-domain features.

**Files:**
- `conformer_opt.py`: Optimized Conformer with dual-stream processing (time and frequency domains)
- `EEG_conformer_opt_smote.ipynb`: Training pipeline with SMOTE oversampling
- `EEG_conformer_opt_undersampler.ipynb`: Training pipeline with random undersampling

## Model Architecture

### Standard Conformer
- **Input**: Raw EEG signals (batch, 1, 55 channels, 100 timepoints)
- **Patch Embedding**: CNN-based feature extraction using temporal and spatial convolutions
- **Encoder**: Multi-layer transformer encoder with self-attention mechanism
- **Classification Head**: Fully-connected layers with dropout regularization
- **Output**: Binary classification (remembered vs. forgotten)

### Optimized Conformer
- **Dual-Stream Architecture**: 
  - Time-domain stream: Processes raw EEG signals via patch embedding
  - Frequency-domain stream: Processes band-power features from 5 frequency bands (delta, theta, alpha, beta, gamma)
- **Feature Fusion**: Concatenates time and frequency embeddings before transformer encoding
- **Enhanced Representation**: Leverages both temporal dynamics and spectral characteristics

## Data Processing

Both implementations support two class balancing strategies:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples for the minority class
2. **Random Undersampling**: Reduces majority class samples to balance dataset

**Preprocessing Pipeline:**
- Channel selection (removes artifact-prone channels)
- Spatial filtering (Common Average Reference)
- Bandpass filtering (0.5-45 Hz)
- Standardization (z-score normalization)

## Training Configuration

- **Optimizer**: Adam with learning rate scheduling (ReduceLROnPlateau)
- **Loss Function**: Cross-entropy loss
- **Regularization**: Dropout (0.3-0.5), early stopping
- **Batch Size**: 32
- **Embedding Dimension**: 40
- **Transformer Depth**: 6 layers
- **Attention Heads**: 10

## Usage

### Training Standard Conformer with SMOTE
```python
from conformer.conformer import Conformer
import torch

# Initialize model
model = Conformer(emb_size=40, depth=6, n_classes=2)

# Load data and train (see notebooks for complete pipeline)
# EEG-conformer_wopreprocessing_smote.ipynb
```

### Training Optimized Conformer with Frequency Features
```python
from conformer_optimized.conformer_opt import ConformerWithFreq

# Initialize model
model = ConformerWithFreq(emb_size=40, depth=6, n_classes=2, n_bands=5)

# Requires both time-domain and frequency-domain inputs
# See EEG_conformer_opt_smote.ipynb for implementation
```

## Model Performance

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC

Results are saved with trained model checkpoints (.pt files).

## Data Format

**Expected Input:**
- Chinese dataset: 9 subjects, 63 channels (55 after removal), MATLAB .mat format
- Korean dataset: 14 subjects, 61 channels (55 after removal), HDF5 .mat format
- Variables: `x_post` (EEG data), `label_next_ind` (labels: 0=forgotten, 1=remembered)

## Dependencies

See `requirements.txt` for complete package list.

Key dependencies:
- PyTorch
- einops
- scikit-learn
- imbalanced-learn
- scipy
- h5py

## Citation

If you use this code, please cite the associated publication in Nature Scientific Data.

## Notes

- Models expect input shape: (batch, 1, n_channels, n_timepoints)
- All notebooks contain complete data loading, preprocessing, training, and evaluation pipelines
- Trained model weights are saved as .pt files for inference
