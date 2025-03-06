import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import seaborn as sns
import mne_connectivity
from mne_bids import BIDSPath, read_raw_bids



# %% functions

def update_path(bids_path, sub, task):
    """
    Update BIDSPath for a given subject and task.
    
    Parameters:
        bids_path (BIDSPath): The base BIDS path.
        sub (str): Subject ID.
        task (str): Task name.

    Returns:
        BIDSPath: Updated BIDSPath with subject and task.
    """
    # update the path for the current subject and task
    bids_path.update(subject=sub, task=task, run=None)
    
    # read the parameters of the current path
    parma = bids_path.match()[0]
    run = parma.run
    acc = parma.acquisition

    # update the path with the update run and acquisition
    bids_path.update(run=run, acquisition=acc)

    return bids_path

def plot_and_save(data, ch_names, save_name):
    '''
    Plot and save a heatmap of the functional connectivity matrix.

    Parameters:
        data (numpy.ndarray): Functional connectivity matrix.
        ch_names (list): List of channel names.
        save_name (str): Filename to save the plot.
    '''
    ax_f = sns.heatmap(data, xticklabels=ch_names,
                       yticklabels=ch_names, vmin=0, vmax=1)
    ax_f.set_xticklabels(ch_names, fontsize=5)
    ax_f.set_yticklabels(ch_names, fontsize=3)
    ax_f.get_figure().savefig(save_name)
    plt.clf()    
    
    

def freq_separate(frequency_range, freq_coherence):
    
    if frequency_range[-1] > 250:
        high_gama_end = 250
    else:
        high_gama_end = frequency_range[-1]
        
    freq = {'delta': [0,4] , 'theta': [4, 8], 'alpha': [8,12] , 'beta':[12,30], 
            'low_gamma':[30,70], 'high_gamma':[70,high_gama_end]}
    
    mean_by_bands = []

    for val in freq.values(): 
        s, e = val
        ind_s, ind_e = np.where(frequency_range >= s)[0][0], np.where(frequency_range <= e)[0][-1]
        mean_by_bands.append(np.mean(freq_coherence[ind_s:ind_e]))
        
    return mean_by_bands
 

def functional_connectivity_by_freq_bands(raw):
    """
    Compute functional connectivity using coherence across six frequency bands.

    Parameters:
        raw (mne.io.Raw): Preprocessed MNE raw object containing iEEG data.

    Returns:
        fc (numpy.ndarray): 3D matrix of functional connectivity per frequency band.
        flatten_fc (pandas.DataFrame): Flattened functional connectivity matrix.
    """
    raw_data = raw.get_data()
    len_ch = len(raw.ch_names)
    fc = np.ones((6, len_ch, len_ch))  # diag will stay 1
    fs = raw.info['sfreq']
    
    
    for i in range(len_ch):
        for j in range(i+1, len_ch):  # Only compute upper triangle to avoid redundant calculations (matrix symmetry)
            f, c = scipy.signal.coherence(raw_data[i], raw_data[j], fs, nperseg=fs/2)
            six_bands = freq_separate(f,c)
            fc[:, i, j] = fc[:, j, i] = six_bands
    
    flatten_fc = pd.DataFrame([fc[i,:,:][np.triu_indices(len_ch)] for i in range(6)],
                              index=['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma'])     
    
    return fc, flatten_fc


   
def functional_connectivity(raw, welch):
    """
    Compute functional connectivity using coherence across all frequencies.

    Parameters:
        raw (mne.io.Raw): Preprocessed MNE raw object.
        welch (bool): Whether to use Welch’s method for coherence estimation.

    Returns:
        fc (numpy.ndarray): Functional connectivity matrix.
        flatten_fc (numpy.ndarray): Flattened upper-triangle of the connectivity matrix.
    """
    raw_data = raw.get_data()
    len_ch = len(raw.ch_names)
    fc = np.ones((len_ch, len_ch))  # diag will stay 1
    fs = raw.info['sfreq']

    for i in range(len_ch):
        for j in range(i+1, len_ch):  # Only compute upper triangle to avoid redundant calculations (matrix symmetry)  
            
            # if welch:              
            # else:
            f, c = scipy.signal.coherence(raw_data[i], raw_data[j], fs, nperseg=fs/2)
            fc[i, j] = fc[j, i] = np.mean(c)
    
    flatten_fc = fc[np.triu_indices(len_ch)] 
    
    return fc, flatten_fc


def Preprocessing_mne(raw):
    """
    Preprocess raw MNE data by selecting ECOG channels,
    applying a notch filter, and average re-referencing.

    Parameters:
        raw (mne.io.Raw): Raw iEEG data.

    Returns:
        mne.io.Raw: Preprocessed data.
    """     
    # 1. peak only ECOG chnnals & drop noisy channels
    raw.pick_types(ecog=True)
    
    # 2. notch filter for harmonic 50 Hz
    raw.notch_filter(np.arange(50, 251, 50))
    
    # 3. average re-reference
    raw.set_eeg_reference()
    
    
    return raw
    
def process_subject(sub, bids_path, tasks, filter_by_band):
    """
    Process a single subject across all tasks.
    """
    for task in tasks:
        try:
            process_task(sub, task, bids_path, filter_by_band)
        except (IndexError, ValueError) as error:
            print(f'Error processing subject {sub}, task {task}:', error)
            continue

def process_task(sub, task, bids_path, filter_by_band):
    """
    Process functional connectivity for a single subject and task.
    """
    bids_path = update_path(bids_path, sub, task)
    raw = read_raw_bids(bids_path=bids_path, extra_params=dict(preload=True), verbose=False) # Load bids data to MNE
    raw = Preprocessing_mne(raw)  # Preprocessing
    ch_names = raw.ch_names  # Extract channel names

    if filter_by_band:
        fc, flatten_fc = functional_connectivity_by_freq_bands(raw)
        for i, band in enumerate(freq):
            save_name = f'FC_figures_by_frequency_bands/sub_{sub}_{task}_{band}_coherence.png'
            plot_and_save(fc[i, :, :], ch_names, save_name)
        flatten_fc.to_csv(f'FC_matrix_by_frequency_bands/flatten_sub_{sub}_{task}_coherence.csv')
    else:
        fc, flatten_fc = functional_connectivity(raw)
        pd.DataFrame(fc, index=ch_names, columns=ch_names).to_csv(f'FC_matrix/sub_{sub}_{task}_coherence.csv')
        pd.DataFrame(flatten_fc, index=ch_names, columns=ch_names).to_csv(f'FC_matrix/flatten_sub_{sub}_{task}_coherence.csv')
        plot_and_save(fc, ch_names, f'FC_figures/sub_{sub}_{task}_coherence.png')

def process_subject_time_based(sub, bids_path, tasks, filter_by_band, delta_t=1):
    """
    Process functional connectivity for a single subject across all tasks in time segments.
    """
    for task in tasks:
        try:
            process_task_time_based(sub, task, bids_path, filter_by_band, delta_t)
        except (IndexError, ValueError) as error:
            print(f'Error processing subject {sub}, task {task}:', error)
            continue

def process_task_time_based(sub, task, bids_path, filter_by_band, delta_t):
    """
    Process functional connectivity for a single subject and task over time segments.
    """
    bids_path = update_path(bids_path=bids_path, sub=sub, task=task)
    base_raw = read_raw_bids(bids_path=bids_path, extra_params=dict(preload=True), verbose=False)
    base_raw = Preprocessing_mne(base_raw)
    ch_names = base_raw.ch_names
    flatten_df = pd.DataFrame()

    for i in range(int(base_raw.times[-1])):
        raw = base_raw.copy().crop(i, i + delta_t)
        if filter_by_band:
            fc, flatten_fc = functional_connectivity_by_freq_bands(raw)
            flatten_df = pd.concat([flatten_df, flatten_fc]) if not flatten_df.empty else flatten_fc
        else:
            fc, flatten_fc = functional_connectivity(raw, welch=False)
            flatten_df = flatten_df.append(flatten_fc, ignore_index=True)

    output_path = f'FC_matrix_time_range/all_time_flatten_sub_{sub}_{task}_coherence.csv'
    flatten_df.to_csv(output_path)
    print(f"Saved time-resolved connectivity matrix for {sub}, {task}: {output_path}")
