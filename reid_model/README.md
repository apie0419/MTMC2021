# Vehicle-ReID
Vehicle Re-identification

## Introduction

Vehicle re-identification using AICITY Track 3 dataset.

## Requirements

* Graphics Card: 1080Ti
* System: Ubuntu 18.04
* conda environment: environment.yml
* `conda env create -f environment.yml`


## Execution

### Step 1: Preprocess

1. Get the data from AI City Challenge website: https://www.aicitychallenge.org/2020-data-and-evaluation/
2. `cd ReID/precrocess`
3. Change the path of data in `path.py`
4. Execute `prepare_frame.py`, `extract_vehicle.py` and `extract_test.py` in order


### Step 2: ReID

1. `cd ReID`
2. Change some settings in `options.py`
3. Execute `train_reid.py`
4. You will get the reid model in `checkpoints`