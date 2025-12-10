import mne
import h5py
import scipy.io
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--trials", required=True, type=str,
    help="forgotten, remembered, both")
ap.add_argument("-m", "--method", required=True, type=str,
    help="MNE/dSPM/sLORETA/eLORETA")
args = vars(ap.parse_args())

X_all = []
y_all = []

ch_ch = list(range(63))
g_ch = list(range(61))
ch_rem = [0,2,6,8,29,31,51,55]
g_rem = [46,49,56,58,59,60]

for i in ch_rem:
    ch_ch.remove(i)

for j in g_rem:
    g_ch.remove(j)


print("Chinese load")
for subj in range(1, 10):
    f = scipy.io.loadmat('data_ch/SHOW/sbj_{}.mat'.format(subj))

    X = np.array(f['x_post'])[:-1,ch_ch,:]
    y = np.array(f['label_next_ind'])[0, :]

    X = np.transpose(X)

    X_all.append(X)
    y_all.append(y)

print("German load")
for subj in range(1, 15):
    with h5py.File('data/SHOW/sbj_{}.mat'.format(subj)) as f:

        X = np.array(f['x_post'])[:,g_ch,:]
        y = np.array(f['label_next_ind'])[:, 0]

        X_all.append(X)
        y_all.append(y)

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)

print("X shape: ", X.shape)
print("y shape: ", y.shape)

indices_0 = np.nonzero(y == 0)[0] # forgotten
indices_1 = np.nonzero(y == 1)[0] # remembered

print(indices_0.shape)
print(indices_1.shape)

trials = args['trials']

if trials == 'forgotten':
    X = X[indices_0] # forgotten
elif trials == 'remembered':
    X = X[indices_1] # remembered

print("Final X shape: ", X.shape)

n_channels = 55
sampling_freq = 100		# in Hertz
ch_types = ['eeg'] * 55

ch_names = ['F5', 'F1', 'Fz', 'F2', 'F6', 'FFT7', 'FFT8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'FTT7', 'FCC5', 'FCC6', 'FTT8', 'T7', 'C5', 'C3', 'Cz', 'C4', 'C6', 'T8', 'TTP7', 'TTP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TPP7', 'TPP8', 'P9', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'EOGh', 'EOGv']
g_rem = ['P3','P6','O1','O2','EOGh','EOGv']

for ch in g_rem:
    ch_names.remove(ch)

#print(ch_names)


info = mne.create_info(ch_types=ch_types, ch_names=ch_names, sfreq=sampling_freq)
# info['bads'] = ['EOGv', 'EOGh']


epochs = mne.EpochsArray(X, info)
# epochs.plot(picks='misc', show_scrollbars=True)
# print(epochs[0].info)

from mne.datasets import fetch_fsaverage
import os.path as op

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

montage = mne.channels.make_standard_montage('standard_1005')
epochs.set_montage(montage, on_missing='raise')
epochs.set_eeg_reference(projection=True)  # needed for inverse modeling

# noise_cov = mne.compute_covariance(epochs, method='shrunk', rank=None, verbose=True)
# mne.write_cov('ask_both_all.fif', noise_cov)

noise_cov = mne.read_cov('show_rem_all.fif')
# fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info)

evoked = epochs.average().pick('eeg')
# evoked.plot(time_unit='s')
# evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg',
#                     time_unit='s')

fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)

inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)

method = args['method']
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

# fig, ax = plt.subplots()
# ax.plot(1e3 * stc.times, stc.data[::100, :].T)
# ax.set(xlabel='time (ms)', ylabel='%s value' % method)

# fig, axes = plt.subplots(1, 1)
# evoked.plot(axes=axes)
# axes.texts.clear()
# for line in axes.lines:
#     line.set_color('#98df81')
# residual.plot(axes=axes)

vertno_max_rh, time_max_rh = stc.get_peak(hemi='rh')
vertno_max_lh, time_max_lh = stc.get_peak(hemi='lh')

vertno_max, time_max = stc.get_peak()

subjects_dir = '/home/adlet/mne_data/MNE-fsaverage-data'
surfer_kwargs = dict(background='white',
    hemi='both', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[4.49*(10**(-5)), 6.27*(10**(-5)), 8.03*(10**(-5))]), views='coronal',
    initial_time=min(time_max_rh, time_max_lh), time_unit='s', size=(800, 800), smoothing_steps=10)

# surfer_kwargs = dict(
#     hemi='both', subjects_dir=subjects_dir,
#     clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
#     initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)

brain = stc.plot(**surfer_kwargs)

brain.add_foci(vertno_max_rh, coords_as_verts=True, hemi='rh', color='red',
               scale_factor=0.6, alpha=0.5)
brain.add_foci(vertno_max_lh, coords_as_verts=True, hemi='lh', color='blue',
               scale_factor=0.6, alpha=0.5)

# brain.add_foci(vertno_max, coords_as_verts=True, color='red',
#                scale_factor=0.6, alpha=0.5)


# brain.add_text(0.1, 0.9, f'locations of maximal activation for all subjects \nwith {method} method for {trials} trials', 'title',
               # font_size=10)

input("ENTER to quit")
