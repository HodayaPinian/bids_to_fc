# BIDS iEEG Data to Functional Connectivity

This repository contains Python scripts for computing functional connectivity from intracranial EEG (iEEG) data stored in BIDS format. The analysis uses coherence measures to estimate functional connectivity and includes preprocessing, frequency-specific connectivity analysis, and visualization.

## Data Source
The iEEG dataset used in this project is publicly available on OpenNeuro:
[ds003688 v1.0.7](https://openneuro.org/datasets/ds003688/versions/1.0.7)

## Features
- Reads and processes iEEG data from a BIDS-formatted dataset.
- Preprocesses data (filtering, re-referencing, and channel selection).
- Computes functional connectivity using coherence.
- Supports frequency band-specific connectivity analysis.
- Outputs connectivity matrices and visualizations.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/HodayaPinian/FunctionalConnectivity-BIDS.git
cd FunctionalConnectivity-BIDS
pip install -r requirements.txt
```

## Usage
Run the main script to process the dataset:
```bash
python bids_to_functional_connectivity.py
```

## Dependencies
The required Python packages are listed in `requirements.txt`:
- `mne`
- `mne-bids`
- `numpy`
- `pandas`
- `scipy`
- `openneuro-py`
- `matplotlib`
- `seaborn`

## Outputs
- Functional connectivity matrices (`.csv`)
- Heatmaps (`.png`)

