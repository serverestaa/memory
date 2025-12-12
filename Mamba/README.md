# Mamba-Based Models for EEG-Based Memory Classification

This directory contains implementations of STMambaNet, a spatio-temporal state-space model based on the Mamba architecture for classifying memory retention in EEG data from second language word learning experiments.

## Overview

STMambaNet leverages Mamba state-space models to capture long-range temporal dependencies and spatial patterns in EEG signals. The architecture processes both temporal and spatial dimensions through separate pathways, making it particularly effective for multi-channel time-series EEG data.

## Files

- `Mamba_SMOTE.ipynb`: Training pipeline with SMOTE oversampling for class balancing
- `Mamba_Undersampler.ipynb`: Training pipeline with random undersampling for class balancing

## Model Architecture: STMambaNet

### Overview
STMambaNet is a dual-pathway architecture that separately processes temporal and spatial information from EEG signals using Mamba state-space models.

### Architecture Components

1. **Multi-Scale Temporal Convolutions**
   - Kernel sizes: [5, 9]
   - Channels per kernel: [4, 4]
   - Captures multi-resolution temporal features
   - Total convolutional channels: 8

2. **Spatial Convolution**
   - 1x1 convolution for channel-wise feature mixing
   - Batch normalization and ELU activation
   - Learns spatial relationships between electrodes

3. **Dual Pooling Strategy**
   - **Variance Pooling**: Captures signal variability
   - **Average Pooling**: Captures mean activation patterns
   - Both pathways processed independently

4. **Temporal Pathway (Mamba-T)**
   - Input dimension: 55 (channels)
   - State dimension: 8
   - Layer normalization
   - Residual connections
   - Feedforward network: 55 -> 32 -> 55
   - Dropout: 0.3

5. **Spatial Pathway (Mamba-S)**
   - Input dimension: 100 (timepoints)
   - State dimension: 8
   - Layer normalization
   - Residual connections
   - Feedforward network: 100 -> 64 -> 100
   - Dropout: 0.3

6. **Convolutional Encoders**
   - Temporal encoder: 110 -> 55 channels
   - Spatial encoder: 200 -> 100 channels
   - 1D convolution with batch normalization

7. **Feature Fusion and Classification**
   - Spatial features: 55×100 -> 32 (FC layer)
   - Temporal features: 100×55 -> 32 (FC layer)
   - Concatenated features: 64 dimensions
   - Final classification: 64 -> 2 classes (softmax)

### Input/Output Specifications

**Input Shape**: (batch, 55 channels, 100 timepoints)
**Output Shape**: (batch, 2 classes)
**Output**: Binary classification probabilities (remembered vs. forgotten)

## Data Processing Pipeline

### 1. Data Loading
- **Chinese Dataset**: 9 subjects, MATLAB .mat format
- **Korean Dataset**: 14 subjects, HDF5 .mat format
- Channel selection reduces to 55 channels

### 2. Advanced Preprocessing

**Spatial Filtering:**
- Common Average Reference (CAR)
- Removes global artifacts while preserving local activity

**Temporal Filtering:**
- Bandpass: 0.5-45 Hz (4th order Butterworth)
- Removes drift and high-frequency noise
- Processed trial-by-trial for memory efficiency

**Normalization:**
- Robust normalization using Median Absolute Deviation (MAD)
- Formula: (data - median) / (1.4826 × MAD)
- Less sensitive to outliers than z-score
- Preserves memory-related amplitude differences

### 3. Class Balancing

**SMOTE Approach:**
- Synthetic Minority Over-sampling Technique
- Generates synthetic samples in feature space
- Applied to training set only

**Undersampling Approach:**
- Random undersampling of majority class
- Balances dataset while preserving minority samples

### 4. Data Splitting
- Training: 80%
- Validation: 10%
- Test: 10%
- Random state: 42

## Training Configuration

### Optimizer
- **Adam optimizer**
- Learning rate: 0.001
- Weight decay: 1e-5

### Learning Rate Scheduling
- ReduceLROnPlateau
- Patience: 5 epochs
- Factor: 0.5
- Minimum learning rate: 1e-6

### Loss Function
- Cross-entropy loss
- Class weights for balanced training

### Regularization
- Dropout: 0.3
- Early stopping: patience 10 epochs
- Gradient clipping: max norm 1.0

### Training Parameters
- Batch size: 32
- Maximum epochs: 100
- Early stopping based on validation loss

## Key Features of Mamba Architecture

### Advantages over Transformers

1. **Linear Complexity**: O(n) vs O(n²) for self-attention
2. **Efficient Long Sequences**: Better scaling for long EEG sequences
3. **State-Space Modeling**: Captures continuous dynamics
4. **Hardware Efficient**: Optimized for GPU computation

### State-Space Model

The Mamba block implements selective state-space models:

```
h(t) = A·h(t-1) + B·x(t)
y(t) = C·h(t) + D·x(t)
```

Where:
- h(t): Hidden state
- x(t): Input
- y(t): Output
- A, B, C, D: Learned parameters

## Usage

### Installation
```bash
pip install ninja cmake
pip install torch torchvision torchaudio
pip install --no-build-isolation mamba-ssm
```

### Training with SMOTE
```python
from mamba_ssm import Mamba
import torch
from imblearn.over_sampling import SMOTE

# Load and preprocess data
processor = EnhancedEEGProcessor(base_path='path/to/data')
processor.load_chinese_data(condition='SHOW')
processor.load_korean_data(condition='SHOW')

# Apply preprocessing
X_filtered = processor.apply_bandpass_filter(X_raw)
X_normalized = processor.apply_memory_normalization(X_filtered)

# Apply SMOTE
smt = SMOTE(random_state=42)
X_train, y_train = smt.fit_resample(X_train_flat, y_train)

# Initialize model
model = STMambaNet(input_size=55, hidden_size=55, num_classes=2)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training loop
for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)
```

### Inference
```python
# Load trained model
model = STMambaNet(input_size=55, hidden_size=55, num_classes=2)
model.load_state_dict(torch.load('best_model_mamba.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_test)
    pred_classes = torch.argmax(predictions, dim=1)
```

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Class-specific precision
- **Recall**: Class-specific recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Classification breakdown
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Memory Optimization

Both notebooks implement memory-efficient processing:

1. **Trial-by-Trial Filtering**: Processes one trial at a time
2. **Float32 Precision**: Reduces memory footprint
3. **Garbage Collection**: Explicit memory cleanup
4. **Batch Processing**: Configurable batch sizes
5. **Progress Tracking**: Monitors processing steps

## Model Advantages

1. **Long-Range Dependencies**: Mamba excels at capturing temporal patterns
2. **Dual-Pathway Design**: Separately models temporal and spatial information
3. **Multi-Scale Features**: Combines variance and average pooling
4. **Efficient Computation**: Linear complexity enables fast training
5. **Robust Normalization**: MAD-based scaling preserves memory signals

## Implementation Notes

### Hardware Requirements
- GPU recommended (CUDA support)
- Minimum 8GB GPU memory
- CPU mode supported but slower

### Compatibility
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- Python 3.8+

## Dependencies

See `requirements.txt` for complete package list.

Key dependencies:
- PyTorch
- mamba-ssm
- scikit-learn
- imbalanced-learn
- scipy
- h5py

## Citation

If you use this code, please cite the associated publication in Nature Scientific Data.

## Notes

- Mamba state-space models are particularly effective for long EEG sequences
- Dual-pathway architecture captures complementary temporal and spatial features
- Robust normalization is critical for preserving memory-related signals
- Both SMOTE and undersampling approaches are provided for class balancing
- Pre-trained weights can be saved and loaded for inference
- Model supports both classification and feature extraction modes
