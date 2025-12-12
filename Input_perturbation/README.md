# Input Perturbation Analysis for EEG-Based Memory Classification

This directory contains implementations for analyzing model interpretability through input perturbation techniques and topographic mapping of EEG signals in memory retention classification tasks.

## Overview

Input perturbation analysis identifies which EEG channels and temporal features are most critical for model predictions. This approach provides insights into the neural mechanisms underlying memory encoding during second language word learning.

## Files

### Input Perturbation Analysis
- `InputPerturbation_StmambaNet_CorrectAndWrongPredictions_TrainDataSet.ipynb`: Analyzes STMambaNet predictions on training data, comparing correctly and incorrectly classified trials
- `InputPerturbation_StmambaNet_CorrectAndWrongPredictions_TrainDataSet (1).ipynb`: Alternative version of perturbation analysis

### Topographic Mapping
- `Topomaps_ShowTrials_InputPerturbation_StmambaNet_TrainingData_ForUpload.ipynb`: Generates topographic maps showing channel importance for SHOW trials
- `Topomaps_AskTrials_InputPerturbation_StmambaNet_TrainingData_ForUpload.ipynb`: Generates topographic maps showing channel importance for ASK trials

## Methodology

### Input Perturbation Technique

The perturbation analysis identifies critical features by systematically masking or altering input components and measuring prediction changes.

**Process:**
1. **Baseline Prediction**: Obtain model prediction on unperturbed EEG trial
2. **Channel Perturbation**: Individually zero out each of 55 EEG channels
3. **Prediction Comparison**: Measure change in prediction confidence
4. **Importance Scoring**: Channels causing largest prediction drops are most important

**Perturbation Strategy:**
- Zero-out perturbation: Set channel values to zero
- Noise perturbation: Replace channel with Gaussian noise
- Shuffle perturbation: Randomize temporal order

### Analysis Categories

1. **Correct Predictions**
   - Trials correctly classified as remembered or forgotten
   - Identifies features consistently supporting accurate predictions

2. **Incorrect Predictions**
   - Trials misclassified by the model
   - Reveals features leading to errors
   - Helps understand model limitations

### Topographic Mapping

**Visualization:**
- 2D scalp topography showing channel importance
- Color-coded heatmaps (warmer colors = higher importance)
- Spatial distribution of critical brain regions

**Trial Types:**
- **SHOW Trials**: Passive viewing of word-picture pairs
- **ASK Trials**: Active recall testing

## Implementation Details

### Data Processing Pipeline

1. **Data Loading**
   - Chinese dataset: 9 subjects, 55 channels
   - Korean dataset: 14 subjects, 55 channels
   - Trial labels: 0 (forgotten), 1 (remembered)

2. **Preprocessing**
   - Spatial filtering: Common Average Reference (CAR)
   - Bandpass filtering: 0.5-45 Hz
   - Notch filtering: 50 Hz (power line noise)
   - Robust normalization: Median-based scaling

3. **Frequency Feature Extraction**
   - Delta: 0.5-4 Hz
   - Theta: 4-8 Hz
   - Alpha: 8-13 Hz
   - Beta: 13-30 Hz
   - Gamma: 30-45 Hz

### Model: STMambaNet

**Architecture:**
- Spatio-temporal processing with Mamba state-space model
- Multi-scale temporal convolutions (kernels: 5, 9)
- Dual pathways: variance and average pooling
- Separate temporal and spatial Mamba blocks

**Input:** (batch, 55 channels, 100 timepoints)
**Output:** Binary classification (remembered vs. forgotten)

### Perturbation Analysis Workflow

```python
# Load pre-trained STMambaNet
model = STMambaNet(input_size=55, hidden_size=55, num_classes=2)
model.load_state_dict(torch.load('best_model_mamba.pt'))
model.eval()

# For each trial
for trial in test_data:
    # Baseline prediction
    baseline_pred = model(trial)
    
    # Perturb each channel
    importance_scores = []
    for ch in range(55):
        perturbed_trial = trial.clone()
        perturbed_trial[:, ch, :] = 0  # Zero out channel
        
        perturbed_pred = model(perturbed_trial)
        importance = abs(baseline_pred - perturbed_pred)
        importance_scores.append(importance)
    
    # Identify most important channels
    top_channels = np.argsort(importance_scores)[-10:]
```

### Topographic Map Generation

```python
import matplotlib.pyplot as plt
from mne.viz import plot_topomap

# Average importance across trials
avg_importance = np.mean(channel_importance_matrix, axis=0)

# Create topomap
plot_topomap(avg_importance, 
             pos=channel_positions,
             show=True,
             cmap='RdYlBu_r',
             contours=6)
```

## Key Findings

### Expected Results

1. **Frontal Regions**: Important for encoding and working memory
2. **Parietal Regions**: Critical for retrieval and attention
3. **Temporal Regions**: Language processing and semantic memory
4. **Occipital Regions**: Visual processing of stimuli

### Comparison: Correct vs. Incorrect Predictions

- **Correct Predictions**: Consistent activation patterns across frontal-parietal networks
- **Incorrect Predictions**: Noisy or atypical activation patterns

### SHOW vs. ASK Trials

- **SHOW**: Stronger visual cortex involvement during encoding
- **ASK**: Enhanced frontal-parietal activation during retrieval

## Visualization Outputs

1. **Channel Importance Heatmaps**
   - Spatial distribution of channel importance
   - Separate maps for correct/incorrect predictions

2. **Time-Resolved Importance**
   - Temporal dynamics of feature importance
   - Critical time windows for memory classification

3. **Frequency Band Analysis**
   - Band-specific importance (delta, theta, alpha, beta, gamma)
   - Identifies frequency-specific memory markers

## Usage

### Running Perturbation Analysis
```python
# Open notebook
# InputPerturbation_StmambaNet_CorrectAndWrongPredictions_TrainDataSet.ipynb

# Load data and model
processor = EnhancedEEGProcessor(base_path='path/to/data')
processor.load_chinese_data(condition='SHOW')
processor.load_korean_data(condition='SHOW')

# Run perturbation analysis
correct_importances, incorrect_importances = analyze_predictions()

# Visualize results
plot_channel_importance(correct_importances, title='Correct Predictions')
plot_channel_importance(incorrect_importances, title='Incorrect Predictions')
```

### Generating Topomaps
```python
# Open notebook
# Topomaps_ShowTrials_InputPerturbation_StmambaNet_TrainingData_ForUpload.ipynb

# Compute channel importance
importance_matrix = compute_channel_importance()

# Generate topographic maps
plot_topomap_comparison(importance_matrix)
```

## Applications

1. **Model Interpretability**: Understand which brain regions drive predictions
2. **Feature Selection**: Identify most informative channels for dimension reduction
3. **Neuroscience Insights**: Validate neural correlates of memory encoding
4. **Clinical Applications**: Identify biomarkers for memory disorders

## Dependencies

See `requirements.txt` for complete package list.

Key dependencies:
- PyTorch
- mamba-ssm
- MNE-Python (for topographic plotting)
- scikit-learn
- matplotlib
- seaborn

## Citation

If you use this code, please cite the associated publication in Nature Scientific Data.

## Notes

- Perturbation analysis is computationally intensive (requires multiple forward passes per trial)
- Results are averaged across multiple trials for robust importance estimates
- Topomaps require standard 10-20 electrode position system
- Compatible with both Chinese and Korean datasets
- Analysis can be extended to other model architectures (Conformer, EEGNet)
