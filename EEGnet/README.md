# EEGNet for EEG-Based Memory Classification

This directory contains EEGNet implementations for classifying memory retention in EEG data from second language word learning experiments.

## Overview

EEGNet is a compact convolutional neural network specifically designed for EEG-based brain-computer interfaces. This implementation classifies whether words were remembered or forgotten during second language acquisition tasks.

## Files

- `EEGNetGeneral_SMOTE.ipynb`: Training pipeline with SMOTE oversampling for class balancing
- `EEGNetGeneral_undersampler.ipynb`: Training pipeline with random undersampling for class balancing

## Model Architecture

### EEGNet Structure
1. **Temporal Convolution**: 8 filters, kernel size (1, 32)
   - Learns temporal features across time dimension
   - Batch normalization and ELU activation
   - Dropout (0.25)

2. **Depthwise Convolution**: (55, 1) kernel, depth multiplier 2
   - Spatial filtering across all EEG channels
   - Models spatial relationships between electrodes
   - Batch normalization and ELU activation
   - Dropout (0.25)

3. **Separable Convolution**: 16 filters, kernel size (1, 16)
   - Further temporal feature extraction
   - Reduces computational complexity
   - Batch normalization and ELU activation
   - Dropout (0.25)

4. **Global Average Pooling**: Reduces spatial dimensions

5. **Dense Classification Layer**: 2 units with softmax activation
   - Binary classification (remembered vs. forgotten)

**Input Shape**: (55 channels, 100 timepoints, 1)
**Output**: Binary classification probabilities

## Data Processing Pipeline

### 1. Data Loading
- **Chinese Dataset**: 9 subjects, 63 channels (55 after removal)
  - Format: MATLAB .mat files
  - Variables: `x_post`, `label_next_ind`
- **Korean Dataset**: 14 subjects, 61 channels (55 after removal)
  - Format: HDF5 .mat files

### 2. Channel Selection
- Removes artifact-prone channels
- Chinese: removes channels [0, 2, 6, 8, 29, 31, 51, 55]
- Korean: removes channels [46, 49, 56, 58, 59, 60]
- Final channel count: 55 channels

### 3. Preprocessing
- Flattening and reshaping: (trials, channels, timepoints)
- Standardization: z-score normalization using StandardScaler
- Reshaping for CNN input: (trials, channels, timepoints, 1)

### 4. Class Balancing

**SMOTE Approach:**
- Synthetic Minority Over-sampling Technique
- Generates synthetic samples for minority class
- Applied only to training set
- Random state: 42

**Undersampling Approach:**
- Random undersampling of majority class
- Balances dataset by reducing majority samples
- Preserves all minority class samples

### 5. Data Splitting
- Training: 80%
- Validation: 10% (from test split)
- Test: 10%
- Random state: 42

## Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Sparse categorical cross-entropy
- **Batch Size**: 32
- **Epochs**: 10
- **Metrics**: Accuracy

## Usage

### Training with SMOTE
```python
from keras.models import Model
from imblearn.over_sampling import SMOTE

# Load and preprocess data
# Apply SMOTE to training set
smt = SMOTE(random_state=42)
X_train, y_train = smt.fit_resample(X_train_flattened, y_train)

# Train model
model = EEGNet(input_shape=(55, 100, 1), num_classes=2)
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=10, batch_size=32)
```

### Training with Undersampling
```python
from imblearn.under_sampling import RandomUnderSampler

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train_flattened, y_train)

# Train model (same configuration as SMOTE)
```

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy on test set
- **Training Curves**: Accuracy and loss across epochs
- **Visualization**: Training vs. validation performance plots

## Model Output

- **Predictions**: Softmax probabilities for each class
- **Test Accuracy**: Final performance on held-out test set
- **Training History**: Saved for performance analysis

## Advantages of EEGNet

1. **Compact Architecture**: Few parameters, prevents overfitting
2. **EEG-Specific Design**: Tailored for temporal and spatial EEG characteristics
3. **Efficient Training**: Fast convergence, suitable for limited data
4. **Interpretable**: Clear separation of temporal and spatial feature learning

## Dependencies

See `requirements.txt` for complete package list.

Key dependencies:
- TensorFlow/Keras
- scikit-learn
- imbalanced-learn
- scipy
- h5py

## Citation

If you use this code, please cite the associated publication in Nature Scientific Data.

## Notes

- Both notebooks contain complete end-to-end pipelines
- Models are evaluated on balanced test sets
- Performance visualizations included in notebooks
- Compatible with both Chinese and Korean datasets
