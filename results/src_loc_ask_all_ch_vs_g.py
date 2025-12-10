import mne
import h5py
import scipy.io
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse
import matplotlib.pyplot as plt
import argparse

# Script 1: All trials (Remembered + Forgotten) comparison

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", default="MNE", type=str,
    help="MNE/dSPM/sLORETA/eLORETA")
args = vars(ap.parse_args())

X_all_ch = []
y_all_ch = []

X_all_g = []
y_all_g = []

# Common channels between Chinese and German datasets
ch_ch = list(range(63))
g_ch = list(range(61))
ch_rem = [0,2,6,8,29,31,51,55]  # Channels to remove from Chinese
g_rem = [46,49,56,58,59,60]  # Channels to remove from German

for i in ch_rem:
    ch_ch.remove(i)

for j in g_rem:
    g_ch.remove(j)

print("Loading Chinese ASK data (All trials)...")
for subj in range(1, 10):
    f = scipy.io.loadmat('data_ch/ASK/sbj_{}.mat'.format(subj))
    X = np.array(f['x_post'])[:-1, ch_ch, :]
    y = np.array(f['label_next_ind'])[0, :]
    X = np.transpose(X)
    X_all_ch.append(X)
    y_all_ch.append(y)

print("Loading German ASK data (All trials)...")
for subj in range(1, 15):
    with h5py.File('data/ASK/sbj_{}.mat'.format(subj)) as f:
        X = np.array(f['x_post'])[:, g_ch, :]
        y = np.array(f['label_next_ind'])[:, 0]
        X_all_g.append(X)
        y_all_g.append(y)

X_ch = np.concatenate(X_all_ch, axis=0)
y_ch = np.concatenate(y_all_ch, axis=0)

X_g = np.concatenate(X_all_g, axis=0)
y_g = np.concatenate(y_all_g, axis=0)

print(f"Chinese - X shape: {X_ch.shape}, y shape: {y_ch.shape}")
print(f"German - X shape: {X_g.shape}, y shape: {y_g.shape}")

# Setup MNE info
n_channels = 55
sampling_freq = 100
ch_types = ['eeg'] * 55

ch_names = ['F5', 'F1', 'Fz', 'F2', 'F6', 'FFT7', 'FFT8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'FTT7', 'FCC5', 'FCC6', 'FTT8', 'T7', 'C5', 'C3', 'Cz', 'C4', 'C6', 'T8', 'TTP7', 'TTP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TPP7', 'TPP8', 'P9', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'EOGh', 'EOGv']
g_rem_names = ['P3', 'P6', 'O1', 'O2', 'EOGh', 'EOGv']

for ch in g_rem_names:
    ch_names.remove(ch)

info = mne.create_info(ch_types=ch_types, ch_names=ch_names, sfreq=sampling_freq)

from mne.datasets import fetch_fsaverage
import os.path as op

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

subject = 'fsaverage'
trans = 'fsaverage'
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

montage = mne.channels.make_standard_montage('standard_1005')

# Process Chinese data
print("\nProcessing Chinese data...")
epochs_ch = mne.EpochsArray(X_ch, info)
epochs_ch.set_montage(montage, on_missing='raise')
epochs_ch.set_eeg_reference(projection=True)

noise_cov_ch = mne.compute_covariance(epochs_ch, method=['shrunk', 'empirical'], rank=None, verbose=True)
evoked_ch = epochs_ch.average().pick('eeg')

fwd_ch = mne.make_forward_solution(epochs_ch.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)
inverse_operator_ch = make_inverse_operator(evoked_ch.info, fwd_ch, noise_cov_ch, loose=0.2, depth=0.8)

method = args['method']
snr = 3.
lambda2 = 1. / snr ** 2
stc_ch, residual_ch = apply_inverse(evoked_ch, inverse_operator_ch, lambda2,
                                    method=method, pick_ori=None,
                                    return_residual=True, verbose=True)

# Process German data
print("\nProcessing German data...")
epochs_g = mne.EpochsArray(X_g, info)
epochs_g.set_montage(montage, on_missing='raise')
epochs_g.set_eeg_reference(projection=True)

noise_cov_g = mne.compute_covariance(epochs_g, method=['shrunk', 'empirical'], rank=None, verbose=True)
evoked_g = epochs_g.average().pick('eeg')

fwd_g = mne.make_forward_solution(epochs_g.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)
inverse_operator_g = make_inverse_operator(evoked_g.info, fwd_g, noise_cov_g, loose=0.2, depth=0.8)

stc_g, residual_g = apply_inverse(evoked_g, inverse_operator_g, lambda2,
                                  method=method, pick_ori=None,
                                  return_residual=True, verbose=True)

# Calculate difference: Chinese - German
print("\nCalculating Chinese - German difference...")
stc_diff = stc_ch - stc_g

vertno_max_rh_ch, time_max_rh_ch = stc_ch.get_peak(hemi='rh')
vertno_max_lh_ch, time_max_lh_ch = stc_ch.get_peak(hemi='lh')

vertno_max_rh_g, time_max_rh_g = stc_g.get_peak(hemi='rh')
vertno_max_lh_g, time_max_lh_g = stc_g.get_peak(hemi='lh')

# Use the correct subjects_dir from fsaverage download
subjects_dir = op.dirname(fs_dir)

# Plot Chinese
print("\nPlotting Chinese brain...")
surfer_kwargs = dict(background='white',
    hemi='both', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='coronal',
    initial_time=min(time_max_rh_ch, time_max_lh_ch), time_unit='s', size=(800, 800), smoothing_steps=10)

brain_ch = stc_ch.plot(**surfer_kwargs)
brain_ch.add_foci(vertno_max_rh_ch, coords_as_verts=True, hemi='rh', color='red',
                  scale_factor=0.6, alpha=0.5)
brain_ch.add_foci(vertno_max_lh_ch, coords_as_verts=True, hemi='lh', color='blue',
                  scale_factor=0.6, alpha=0.5)
brain_ch.add_text(0.1, 0.9, f'Chinese ASK - All trials\nwith {method} method', 'title', font_size=10)

input("Press ENTER to continue to German...")
try:
    brain_ch.close()
except (AttributeError, RuntimeError):
    pass  # Window already closed

# Plot German
print("\nPlotting German brain...")
surfer_kwargs = dict(background='white',
    hemi='both', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='coronal',
    initial_time=min(time_max_rh_g, time_max_lh_g), time_unit='s', size=(800, 800), smoothing_steps=10)

brain_g = stc_g.plot(**surfer_kwargs)
brain_g.add_foci(vertno_max_rh_g, coords_as_verts=True, hemi='rh', color='red',
                 scale_factor=0.6, alpha=0.5)
brain_g.add_foci(vertno_max_lh_g, coords_as_verts=True, hemi='lh', color='blue',
                 scale_factor=0.6, alpha=0.5)
brain_g.add_text(0.1, 0.9, f'German ASK - All trials\nwith {method} method', 'title', font_size=10)

input("Press ENTER to continue to difference...")
try:
    brain_g.close()
except (AttributeError, RuntimeError):
    pass  # Window already closed

