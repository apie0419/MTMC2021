# TSN

![](https://github.com/apie0419/MTMC2021/blob/master/figures/TSN.png)

# Data Preparation

First of all, write your own configuration in `config/config.yaml` and `mct_mode/config/config.yaml`.

## Training Data

1. run `pipeline/crop_gt_images.py` with `TRAIN_PATH` set to the path of your training dataset.
2. run `pipeline/prepare_tracklets_data.py` with `INPUT_DIR` set to the path of your training dataset.
3. run `mct_model/prepare_data.py` with INPUT_DIR set `path` to the path of your training dataset.

## Validation Data

Same as Training Data, just set those variables to the path of your validation dataset.

## Train the model

run `mct_model/train.py -o <training_log_filename>`

## Validation

1. set `weight` in `mct_model/exp.sh` to the path of your TSN model weight.
2. run `mct_model/exp.sh` to obtain the IDF1 score of the validation set.