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
from mne_bids import BIDSPath
from function_bids_to_fc import process_subject, process_subject_time_based  
import openneuro

# # to dowenload first time the data:
# dataset ='ds003688'
# bids_root = /.... # Change path accordingly
# openneuro.download(dataset=dataset, target_dir=bids_root)


# for next time 
# Define BIDS root directory
bids_root = /....  # Change path accordingly
os.chdir(path=bids_root)

# Load participant metadata
participants = pd.read_table('participants.tsv')

# Filter to analyze only iEEG subjects
ieeg_subjects = participants['participant_id'][participants['iEEG'] == 'yes'].str.replace("sub-", "", regex=True)
print(ieeg_subjects)

# BIDS parameters
session = 'iemu'
datatype = 'ieeg'
acquisition = 'clinical'
suffix = 'ieeg'

tasks = ['film', 'rest']
filter_by_band = True
freq = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']

# the ground bids path for all subjects
base_bids_path = BIDSPath(root=bids_root, session=session,
                     datatype=datatype, suffix=suffix)


# Main Loop for Static Functional Connectivity
for sub in ieeg_subjects:
    process_subject(sub=sub, bids_path=base_bids_path, tasks=tasks, filter_by_band=filter_by_band)


# Main Loop for Time-Based Functional Connectivity Analysis - 1 second time segments
for sub in ieeg_subjects:
    print(f"Processing subject {sub}")
    process_subject_time_based(sub=sub, bids_path=base_bids_path, tasks=tasks, filter_by_band=filter_by_band)
