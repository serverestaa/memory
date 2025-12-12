# Source Localization Analysis for EEG-Based Memory Studies

This directory contains scripts for performing EEG source localization analysis to identify cortical regions involved in memory encoding and retrieval during second language word learning tasks.

## Overview

Source localization transforms scalp-recorded EEG signals into estimates of underlying brain activity. These analyses identify which cortical regions show differential activation for remembered vs. forgotten words, providing insights into the neural mechanisms of memory formation.

## Files

### Individual Subject Analysis

#### SHOW Trials (Encoding Phase)
- `src_loc_show_all_ch_vs_g.py`: All trials (Chinese dataset)
- `src_loc_show_remembered_ch_vs_g.py`: Remembered trials only
- `src_loc_show_forgotten_ch_vs_g.py`: Forgotten trials only

#### ASK Trials (Retrieval Phase)
- `src_loc_ask_all_ch_vs_g.py`: All trials (Chinese dataset)
- `src_loc_ask_remembered_ch_vs_g.py`: Remembered trials only
- `src_loc_ask_forgotten_ch_vs_g.py`: Forgotten trials only

### General Analysis Scripts
- `src_localization.py`: Basic source localization pipeline for single subject
- `src_localization_ch.py`: Chinese dataset specific processing
- `src_localization_all_ch.py`: All Chinese subjects analysis
- `src_localization_all.py`: Cross-subject analysis
- `src_localization_all_both.py`: Combined Chinese and Korean datasets

## Methodology

### Source Localization Pipeline

The analysis follows standard MNE-Python procedures for distributed source estimation:

1. **Data Preparation**
   - Load EEG epochs from HDF5/MATLAB files
   - Channel configuration: 61 channels (59 EEG + 2 EOG)
   - Sampling frequency: 100 Hz
   - Standard 10-05 electrode montage

2. **Forward Modeling**
   - Head model: fsaverage template brain
   - Boundary Element Method (BEM) model
   - Source space: ico-5 (5120 sources per hemisphere)
   - EEG reference: Average reference projection

3. **Inverse Solution Methods**

   Available methods (specified via command-line argument):
   
   - **MNE (Minimum Norm Estimate)**: L2-norm regularized solution
   - **dSPM (Dynamic Statistical Parametric Mapping)**: Noise-normalized MNE
   - **sLORETA (Standardized Low-Resolution Electromagnetic Tomography)**: Standardized solution with zero localization bias
   - **eLORETA (Exact Low-Resolution Electromagnetic Tomography)**: Improved localization accuracy

4. **Noise Covariance Estimation**
   - Methods: Shrinkage estimator and empirical covariance
   - Computed from baseline or entire epoch
   - Rank: Automatic rank estimation

5. **Source Reconstruction**
   - SNR: 3.0
   - Regularization: λ² = 1/SNR²
   - Loose orientation constraint: 0.2
   - Depth weighting: 0.8

## Channel Configuration

### Standard 10-05 Montage (59 EEG Channels)
```
Frontal: F5, F1, Fz, F2, F6, FFT7, FFT8, FT9, FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8, FT10
Fronto-Temporal: FTT7, FCC5, FCC6, FTT8
Central: T7, C5, C3, Cz, C4, C6, T8
Centro-Parietal: TTP7, TTP8, TP9, TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8, TP10
Parietal: TPP7, TPP8, P9, P5, P3, Pz, P4, P6, P10
Parieto-Occipital: PO7, PO3, POz, PO4, PO8
Occipital: O1, Oz, O2
EOG: EOGh, EOGv (marked as bad channels)
```

## Usage

### Command-Line Interface

```bash
python src_localization.py \
    --subject 1 \
    --trials remembered \
    --method dSPM
```

**Parameters:**
- `--subject`: Subject ID (1-14 for Korean, 1-9 for Chinese)
- `--trials`: Trial type (all, remembered, forgotten, both)
- `--method`: Inverse method (MNE, dSPM, sLORETA, eLORETA)

### Examples

**Analyze remembered trials for subject 5 using dSPM:**
```bash
python src_localization.py --subject 5 --trials remembered --method dSPM
```

**Analyze all trials for subject 3 using sLORETA:**
```bash
python src_localization.py --subject 3 --trials all --method sLORETA
```

**Compare Chinese and Korean datasets:**
```bash
python src_loc_show_all_ch_vs_g.py --subject 1 --method dSPM
```

## Analysis Workflow

### 1. Load EEG Data
```python
import h5py
import mne

# Load data for subject
with h5py.File('data/SHOW/sbj_1.mat') as f:
    X = np.array(f['x_post'])  # (trials, 61 channels, 500 timepoints)
    y = np.array(f['label_next_ind'])[:, 0]  # Labels

# Filter by trial type
indices_remembered = np.nonzero(y == 1)[0]
X_remembered = X[indices_remembered]
```

### 2. Create MNE Epochs Object
```python
# Create info structure
info = mne.create_info(ch_types=ch_types, ch_names=ch_names, sfreq=100)
info['bads'] = ['EOGv', 'EOGh']

# Create epochs
epochs = mne.EpochsArray(X, info)
epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs.set_eeg_reference(projection=True)
```

### 3. Compute Forward Solution
```python
from mne.datasets import fetch_fsaverage

# Load fsaverage template
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# Forward model
fwd = mne.make_forward_solution(
    epochs.info, 
    trans='fsaverage', 
    src=src, 
    bem=bem, 
    eeg=True, 
    mindist=5.0
)
```

### 4. Compute Inverse Solution
```python
from mne.minimum_norm import make_inverse_operator, apply_inverse

# Noise covariance
noise_cov = mne.compute_covariance(
    epochs, 
    method=['shrunk', 'empirical']
)

# Average evoked response
evoked = epochs.average().pick('eeg')

# Inverse operator
inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, 
    loose=0.2, depth=0.8
)

# Apply inverse
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse(
    evoked, inverse_operator, lambda2,
    method='dSPM', pick_ori=None
)
```

### 5. Visualize Results
```python
# Find peaks
vertno_max_rh, time_max_rh = stc.get_peak(hemi='rh')
vertno_max_lh, time_max_lh = stc.get_peak(hemi='lh')

# Plot on brain
brain = stc.plot(
    hemi='both', 
    subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]),
    views='coronal',
    initial_time=min(time_max_rh, time_max_lh),
    smoothing_steps=10
)

# Add foci for maximum activation
brain.add_foci(vertno_max_rh, coords_as_verts=True, 
               hemi='rh', color='red')
brain.add_foci(vertno_max_lh, coords_as_verts=True, 
               hemi='lh', color='blue')
```

## Expected Results

### Remembered Trials
- **Frontal Regions**: Enhanced activation in prefrontal cortex (encoding, working memory)
- **Temporal Regions**: Increased activation in superior/middle temporal gyrus (language processing)
- **Parietal Regions**: Posterior parietal cortex involvement (attention, memory retrieval)

### Forgotten Trials
- **Reduced Frontal Activation**: Weaker prefrontal cortex engagement
- **Lower Temporal Activation**: Reduced language area activity
- **Different Network Patterns**: Altered connectivity patterns

### SHOW vs. ASK Trials
- **SHOW (Encoding)**: Stronger visual cortex and prefrontal activation
- **ASK (Retrieval)**: Enhanced parietal and frontal network activation

## Output Files

Generated visualizations:
- 3D brain surface plots with source activations
- Time course plots of source activity
- Topographic maps at peak activation times
- Residual plots comparing data and model fit

## Analysis Parameters

### Forward Model Settings
- Source space resolution: ico-5 (10,242 sources)
- Minimum distance from inner skull: 5.0 mm
- Conductor model: 3-layer BEM (5120 triangles per layer)

### Inverse Solution Settings
- Loose orientation constraint: 0.2 (mix of fixed and free orientations)
- Depth weighting: 0.8 (compensates for depth bias)
- Regularization: SNR-based (λ² = 1/9)

## Comparison Across Methods

| Method | Characteristics | Best For |
|--------|----------------|----------|
| MNE | Minimum L2-norm, distributed | Smooth, diffuse activations |
| dSPM | Noise-normalized MNE | Statistical comparisons |
| sLORETA | Zero localization error | Deep source localization |
| eLORETA | Exact solution, improved accuracy | Precise localization |

## Dependencies

See `requirements.txt` for complete package list.

Key dependencies:
- MNE-Python
- NumPy
- SciPy
- h5py
- matplotlib

## System Requirements

- MNE-fsaverage data (auto-downloaded)
- Sufficient memory for source space computations (8GB+ recommended)
- 3D visualization requires appropriate graphics backend

## Citation

If you use this code, please cite the associated publication in Nature Scientific Data.

## Notes

- Source localization results are sensitive to head model accuracy
- Individual MRI scans would improve localization precision (fsaverage is a template)
- Multiple inverse methods can be compared for robustness
- Statistical comparison across conditions requires careful consideration of multiple comparisons
- Visualizations require interactive display (not suitable for headless servers)
- Chinese dataset has different channel configurations than Korean dataset
