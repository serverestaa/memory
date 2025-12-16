# EEG-Based Memory Classification: Analysis Code Repository

This repository contains the analysis code used to process and analyze EEG data for memory retention classification in second language word learning experiments. This code accompanies a Data Descriptor submitted to Scientific Data (Nature).

**Important**: This repository contains only the analysis code. The full EEG dataset is hosted in a separate archival repository due to its size.

## Data Availability

The complete EEG dataset associated with this code will be made publicly available in an external data repository. A persistent identifier (DOI) will be provided upon acceptance of the accompanying manuscript.

The dataset comprises:
- EEG recordings from 23 participants (9 Chinese speakers, 14 Korean speakers)
- 61-channel EEG data recorded during second language word learning tasks
- Preprocessed epochs for encoding (SHOW) and retrieval (ASK) phases
- Memory performance labels (remembered vs. forgotten)

## Repository Structure

This repository is organized into the following directories:

### `Conformer/`
Conformer-based neural network architectures for EEG classification.
- **`conformer/`**: Standard Conformer implementation with time-domain processing
  - `conformer.py`: Base model architecture
  - Jupyter notebooks with SMOTE and undersampling strategies
- **`conformer_optimized/`**: Enhanced Conformer with dual-stream (time + frequency) processing
  - `conformer_opt.py`: Optimized model architecture
  - Training pipelines with class balancing methods

### `EEGnet/`
EEGNet compact convolutional neural network implementation.
- `EEGNetGeneral_SMOTE.ipynb`: Training with SMOTE oversampling
- `EEGNetGeneral_undersampler.ipynb`: Training with random undersampling

### `Mamba/`
STMambaNet spatio-temporal state-space models.
- `Mamba_SMOTE.ipynb`: Mamba-based architecture with SMOTE
- `Mamba_Undersampler.ipynb`: Mamba-based architecture with undersampling

### `ensemble/`
Ensemble learning combining multiple deep learning models.
- `ensemble.ipynb`: Majority voting ensemble implementation
- Pre-trained model weights (`.pt` files)
- Model architecture definitions

### `Input_perturbation/`
Model interpretability analysis through input perturbation.
- Perturbation analysis notebooks for identifying critical EEG channels
- Topographic mapping visualization for SHOW and ASK trials

### `source_localization/`
EEG source localization analysis using MNE-Python.
- Scripts for distributed source estimation (dSPM, sLORETA, eLORETA, MNE)
- Separate analyses for encoding and retrieval phases
- Individual and group-level analyses

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU recommended for deep learning models

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/serverestaa/memory.git
cd memory
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies for each analysis module:

**Deep learning models** (Conformer, EEGNet, Mamba, Ensemble):
```bash
pip install -r Conformer/requirements.txt
```

**Source localization**:
```bash
pip install -r source_localization/requirements.txt
```

### Core Dependencies
- PyTorch ≥ 2.0.0
- MNE-Python ≥ 1.4.0
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- scikit-learn ≥ 1.3.0
- imbalanced-learn ≥ 0.11.0
- h5py ≥ 3.9.0
- matplotlib ≥ 3.7.0

See individual `requirements.txt` files in each subdirectory for complete dependency lists.

## Usage

### Data Preparation

After obtaining the dataset from the external repository (DOI will be provided upon manuscript acceptance), place the data files in an appropriate directory and update the file paths in the analysis scripts accordingly.

**Expected data format**:
- Chinese dataset: MATLAB `.mat` files with variables `x_post` (EEG data) and `label_next_ind` (labels)
- Korean dataset: HDF5 `.mat` files with similar structure
- EEG array shape: (n_trials, n_channels, n_timepoints)

### Running Classification Models

Each model directory contains Jupyter notebooks with complete training pipelines:

1. **Conformer models**:
```bash
cd Conformer/conformer
jupyter notebook EEG-conformer_wopreprocessing_smote.ipynb
```

2. **EEGNet**:
```bash
cd EEGnet
jupyter notebook EEGNetGeneral_SMOTE.ipynb
```

3. **Mamba models**:
```bash
cd Mamba
jupyter notebook Mamba_SMOTE.ipynb
```

4. **Ensemble**:
```bash
cd ensemble
jupyter notebook ensemble.ipynb
```

### Running Source Localization

Execute source localization scripts from the command line:

```bash
cd source_localization
python src_loc_show_remembered_ch_vs_g.py --method dSPM
```

Available inverse methods: `dSPM`, `sLORETA`, `eLORETA`, `MNE`

### Running Interpretability Analysis

```bash
cd Input_perturbation
jupyter notebook InputPerturbation_StmambaNet_CorrectAndWrongPredictions_TrainDataSet.ipynb
```

## Reproducibility

This repository represents the exact code version used to generate results presented in the accompanying Data Descriptor manuscript. To ensure reproducibility:

- All analyses were performed using the code at commit hash: `e5a0691`
- Random seeds are set in training scripts for deterministic results
- Model hyperparameters are documented in individual README files within each subdirectory
- Pre-trained model weights are provided in the `ensemble/` directory

To reproduce the exact computational environment, use the provided `requirements.txt` files with the specified version constraints.

## Methods Overview

### Classification Approaches
- **Conformer**: Transformer-based architecture with convolutional patch embedding
- **EEGNet**: Compact CNN designed for EEG-based BCI
- **STMambaNet**: State-space model with dual spatio-temporal pathways
- **Ensemble**: Majority voting across multiple architectures

### Class Balancing Strategies
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random undersampling of majority class

### Preprocessing Pipeline
- Channel selection (55 EEG channels after artifact removal)
- Common Average Reference (CAR)
- Bandpass filtering (0.5–45 Hz)
- Z-score normalization

### Source Localization Methods
- Forward modeling using fsaverage template and BEM
- Inverse solutions: dSPM, sLORETA, eLORETA, MNE
- ico-5 source space (10,240 cortical sources)

## Citation

If you use this code or dataset, please cite the accompanying Data Descriptor:

> [Citation will be provided upon publication]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions regarding this code or the associated dataset, please open an issue on this repository or contact the corresponding author listed in the Data Descriptor manuscript.

## Acknowledgments

This work was supported by [funding information to be added upon publication]. Source localization analyses were performed using MNE-Python. Deep learning models were implemented using PyTorch.
