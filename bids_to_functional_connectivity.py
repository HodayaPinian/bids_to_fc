"""
BIDS iEEG Data to Functional Connectivity with Coherence Measure

This script processes iEEG data stored in BIDS format to compute functional connectivity using coherence.
It performs:
- Data loading from BIDS format
- Preprocessing (filtering, re-referencing)
- Functional connectivity analysis by frequency bands
- Saving connectivity matrices and visualizations

Author: Hodaya Eden Pinian
Date: 2025
"""

import pandas as pd
import os
from mne_bids import BIDSPath, read_raw_bids
import function_bids_to_fc as fc
from function_bids_to_fc import update_path, plot_and_save, functional_connectivity_by_freq_bands, functional_connectivity, Preprocessing_mne
# %%

##
bids_root = r'..\audiovisual_film_dataset' # dir to BIDS data root
os.chdir(bids_root)

participants = pd.read_table('participants.tsv')

# analyze only iEEG subjects
ieeg_subjects = participants['participant_id'][participants['iEEG'] == 'yes']

# basis params for all subjects
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'
task_f = 'film'
task_r = 'rest'

##

# %% functional connectivity for all timepoint in the signal

# create BIDSPath to handel the data
bids_path = BIDSPath(root=bids_root, session=session,
                     datatype=datatype, suffix=suffix)

tasks = [task_f, task_r]
filter_by_band = True
freq = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']


"""
Loop over all subjects and tasks to compute functional connectivity.

- Reads raw iEEG data from BIDS format
- Preprocesses data using MNE functions
- Computes functional connectivity via coherence
- Saves connectivity matrices and plots

Errors are handled for IndexError and ValueError.
"""
for sub in ieeg_subjects[1]:
    for task in tasks:     
        try:
            bids_path = update_path(bids_path, sub, task)

            # Load raw iEEG data from BIDS format   
            raw = read_raw_bids(bids_path=bids_path, extra_params=dict(preload=True), verbose=False)
            
            # Pre - processing
            raw = Preprocessing_mne(raw)
    
            # create functional Connectivity
            ch_names = raw.ch_names # get the channel names and number of channels
            if filter_by_band: # fc by bands:

                fc, flatten_fc = functional_connectivity_by_freq_bands(raw, welch=True)                
                
                # save plt
                for i in range(6):
                    save_name = 'FC_figures_by_frequncy_bands/sub_' + sub +'_' + task + '_' + freq[i] + '_coherence.png'
                    plot_and_save(fc[i,:,:], ch_names, save_name)             
                
                # Save flattened functional connectivity matrix to CSV
                flatten_fc.to_csv('FC_matrix_by_frequncy_bands/flatten_sub_' 
                                  + sub + '_' + task + '_coherence.csv')
                
            else: # fc for all freq conected:

                fc, flatten_fc = functional_connectivity(raw,welch=True)

                # save_matrix 
                pd.DataFrame(fc, index=ch_names, columns=ch_names).to_csv(
                    'FC_matrix/sub_' + sub + '_' + task +'_coherence.csv')

                # Save flattened functional connectivity matrix to CSV
                pd.DataFrame(flatten_fc, index=ch_names, columns=ch_names).to_csv(
                    'FC_matrix/flatten_sub_' + sub + '_' + task +'_coherence.csv')
        
                # plot heatmap
                save_name = 'FC_figures/sub_' + sub +'_' + task + '_coherence.png'
                plot_and_save(fc, ch_names, save_name)

        except IndexError as Error:
            print('Error', Error)
        except ValueError as Error:
            print('Error', Error)
    
            continue



# %% analyze functionl connectivity per delta time in the signal

bids_path = BIDSPath(root=bids_root, session=session,
                     datatype=datatype, suffix=suffix)

delta_t = 1 # time delta in sec
tasks = [task_f, task_r]
filter_by_band = True
freq = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']

"""
Loop over all subjects and tasks to compute functional connectivity.

- Reads raw iEEG data from BIDS format
- Preprocesses data using MNE functions
- Computes functional connectivity via coherence
- Saves connectivity matrices and plots

Errors are handled for IndexError and ValueError.
"""
for sub in ieeg_subjects:
    
    for task in tasks:

        try:
            bids_path = update_path(bids_path, sub, task)
    
            # Load raw iEEG data from BIDS format
            base_raw = read_raw_bids(bids_path=bids_path, extra_params=dict(preload=True), verbose=False)
            
            # Pre - processing
            base_raw = Preprocessing_mne(base_raw)
    
            # Functional Connectivity
            ch_names = base_raw.ch_names
            
            if filter_by_band: # fc by bands:
                
                flatten_df = pd.DataFrame()
                
                for i in range(int(base_raw.times[-1])):

                    raw = base_raw.copy().crop(i, i + delta_t) # crop the data to 1 sec
                                   
                    fc, flatten_fc = functional_connectivity_by_freq_bands(raw)
                    
                    flatten_df = pd.concat([flatten_df, flatten_fc])          
                
                # save flatten
                flatten_df.to_csv('FC_matrix_time_rang_by_frequncy/all_time_flatten_freq_sub_' 
                                  + sub + '_' + task + '_coherence.csv')
                    
            else: # fc for all freq conected:
                flatten_df = []                                
                for i in range(int(base_raw.times[-1])):
                
                    raw = base_raw.copy().crop(i, i+1)
                    
                    fc, flatten_fc = functional_connectivity(raw, welch = False)
                    flatten_df.append(flatten_fc) 
                    
                # save flatten matrix
                pd.DataFrame(flatten_df).to_csv(
                    'FC_matrix_time_range/all_time_flatten_sub_' + sub + '_' + task +'_coherence.csv')


    
        except IndexError as Error:
            print('Error', Error)
        except ValueError as Error:
            print('Error', Error)
    
            continue