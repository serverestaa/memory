# Ensemble Model for EEG-Based Memory Classification

This directory contains an ensemble learning implementation that combines multiple deep learning models for improved classification of memory retention in EEG data from second language word learning experiments.

## Overview

The ensemble approach integrates predictions from three state-of-the-art neural network architectures using majority voting to achieve robust classification performance.

## Files

- `ensemble.ipynb`: Complete ensemble implementation with data loading, model integration, and evaluation
- `conformer.py`: Conformer model architecture (time-domain)
- `conformer_opt.py`: Optimized Conformer with frequency features
- `best_conformer_default.pt`: Pre-trained standard Conformer weights
- `best_conformer_with_freq.pt`: Pre-trained optimized Conformer weights
- `best_model_mamba.pt`: Pre-trained STMambaNet weights

## Ensemble Architecture

### Individual Models

1. **EEG-Conformer (Standard)**
   - Time-domain patch embedding with CNN
   - Transformer encoder with self-attention
   - Embedding size: 40, Depth: 6 layers
   - Model file: `best_conformer_default.pt`

2. **Optimized Conformer with Frequency Features**
   - Dual-stream processing (time + frequency domains)
   - Processes 5 frequency bands (delta, theta, alpha, beta, gamma)
   - Enhanced spectral representation
   - Model file: `best_conformer_with_freq.pt`

3. **STMambaNet**
   - Spatio-temporal Mamba-based architecture
   - Temporal and spatial pathway processing
   - Multi-scale temporal convolutions (kernel sizes: 5, 9)
   - Variance and average pooling branches
   - Mamba state-space model for sequence modeling
   - Model file: `best_model_mamba.pt`

4. **EEGNetv2 (PyTorch Implementation)**
   - Temporal convolution (F1=16 filters)
   - Depthwise spatial convolution (D=2 depth multiplier)
   - Separable convolution (F2=16 filters)
   - Compact and efficient architecture

### Ensemble Strategy

**Majority Voting:**
- Collects predictions from all four models
- Final prediction: most frequent class across models
- Robust to individual model errors
- Improved generalization over single models

## Model Input Requirements

### STMambaNet
- Input shape: (batch, 55 channels, 100 timepoints)
- Multi-scale temporal feature extraction
- Dual-pathway (temporal and spatial) processing

### Conformer Models
- Input shape: (batch, 1, 55 channels, 100 timepoints)
- Conformer-opt requires additional frequency-domain features

### EEGNetv2
- Input shape: (batch, 55 channels, 100 timepoints)
- Automatically adds channel dimension internally

## Data Processing Pipeline

### 1. Data Loading
- **Chinese Dataset**: 9 subjects, MATLAB .mat format
- **Korean Dataset**: 14 subjects, HDF5 .mat format

### 2. Channel Selection
- Chinese: 63 channels to 55 (removes [0, 2, 6, 8, 29, 31, 51, 55])
- Korean: 61 channels to 55 (removes [46, 49, 56, 58, 59, 60])

### 3. Signal Preprocessing
- Spatial filtering: Common Average Reference (CAR)
- Bandpass filtering: 0.5-45 Hz, 4th order Butterworth
- Robust normalization: Median-based (less sensitive to outliers)

### 4. Frequency Feature Extraction (for Conformer-opt)
- Band-power computation for 5 frequency bands
- Delta: 0.5-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-13 Hz
- Beta: 13-30 Hz
- Gamma: 30-45 Hz

### 5. Data Splitting
- Training: 80%
- Validation: 10%
- Test: 10%
- Random state: 42

## Usage

### Loading Pre-trained Models
```python
from ensemble import EnsembleModel
import torch

# Initialize ensemble
ensemble = EnsembleModel(device="cuda" if torch.cuda.is_available() else "cpu")

# Load all models
ensemble.load_models()
```

### Making Predictions
```python
# Prepare test data
X_test = preprocess_eeg_data(raw_data)  # (batch, 55, 100)

# Get ensemble predictions
predictions = ensemble.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
print(f"Ensemble Accuracy: {accuracy:.4f}")
```

### Individual Model Predictions
```python
# Access individual model predictions
pred_conformer = ensemble.models["conformer"](X_test)
pred_stmamba = ensemble.models["stmambanet"](X_test)
pred_conformer_opt = ensemble.models["conformer_opt"](X_test, freq_features)
```

## Configuration

Edit `CONFIG` dictionary in notebook:
```python
CONFIG = {
    "MODEL_PATHS": {
        "conformer": "path/to/best_conformer_default.pt",
        "stmambanet": "path/to/best_model_mamba.pt",
        "conformer_opt": "path/to/best_conformer_with_freq.pt"
    },
    "DATA_PATH_CH": "path/to/chinese/data",
    "DATA_PATH": "path/to/korean/data"
}
```

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Class-specific precision
- **Recall**: Class-specific recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualization of classification results
- **Classification Report**: Detailed per-class metrics

## Advantages of Ensemble Approach

1. **Improved Robustness**: Reduces impact of individual model errors
2. **Complementary Features**: Each model captures different aspects of EEG data
3. **Higher Accuracy**: Typically outperforms individual models
4. **Reduced Overfitting**: Averaging reduces model-specific biases
5. **Diverse Architectures**: Combines CNN, Transformer, and Mamba-based approaches

## Model Specifications

### STMambaNet Details
- Hidden size: 55 (temporal), 100 (spatial)
- Mamba state dimension: 8
- Convolutional channels: [4, 4]
- Kernel sizes: [5, 9]
- Dropout: 0.3

### EEG-Conformer Details
- Embedding dimension: 40
- Transformer layers: 6
- Attention heads: 10
- Dropout: 0.5

### EEGNetv2 Details
- F1 (temporal filters): 16
- D (depth multiplier): 2
- F2 (separable filters): 16
- Kernel length: 10
- Dropout: 0.5

## Dependencies

See `requirements.txt` for complete package list.

Key dependencies:
- PyTorch
- mamba-ssm
- einops
- scikit-learn
- scipy
- h5py

## Citation

If you use this code, please cite the associated publication in Nature Scientific Data.

## Notes

- All models are implemented in PyTorch for framework consistency
- Pre-trained weights are provided for immediate inference
- Ensemble voting can be modified (weighted voting, soft voting)
- Compatible with both Chinese and Korean datasets
- Models are evaluated individually before ensemble combination
