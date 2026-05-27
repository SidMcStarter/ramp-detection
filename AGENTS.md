# Ramp Detection Pipeline Overview

This repository has a three-stage training pipeline plus a prediction notebook:

1. `updated_ditch_latest_version.ipynb`
2. `cnn_dataset_from_remapped_and_lane_csv.ipynb`
3. `pytorch_ramp_cnn_training.ipynb`
4. `pytorch_ramp_cnn_prediction.ipynb`

## Stage 1 - Rule-Based Corridor Processing

`updated_ditch_latest_version.ipynb` is the source-of-truth geometry and rule-based notebook.

It does the following:

- loads trajectory, lane LAS, and sidewalk LAS
- builds station geometry and optional spline-based trajectory evaluation
- optionally refines the trajectory by lane-centre alignment
- extracts lane and sidewalk cross-sections along the corridor
- classifies each station-side with the rule-based ramp / curb / ditch logic
- exports trajectory, sidewalk, and lane point products
- exports structured station-side metrics for later ML use
- optionally supports manual station-side correction and sidewalk remapping

Main outputs from this stage:

- `<base>_sidewalk.csv`
- `<base>_lane.csv`
- `<base>_sidewalk_CC.asc`
- `<base>_lane_CC.asc`
- `<base>_sidewalk_CC.las`
- `<base>_lane_CC.las`
- `<base>_station_side_metrics.csv`
- `<base>_station_side_metrics.json`
- optionally `<base>_station_side_class_corrector.csv`
- optionally `<base>_sidewalk_remapped.csv`

Important ideas:

- `classified` is the main in-memory classified station-side structure
- `meta_R` / `meta_L` contain side-level scalar outputs such as kink `dz` and `slope`
- `dbg_R` / `dbg_L` contain richer debug payloads used for plots and auditing
- `Cell 5.5` exports the structured station-side metrics in a machine-readable format

## Stage 2 - CNN Dataset Creation

`cnn_dataset_from_remapped_and_lane_csv.ipynb` converts one remapped sidewalk CSV plus one exported lane CSV into one ML-ready `cnn_dataset.csv`.

It does the following:

- loads one remapped sidewalk CSV
- loads one lane CSV exported from Stage 1
- loads one station-side metrics CSV exported from Stage 1
- groups data by `(s_m, side)`
- merges sidewalk and lane points into one 1D lateral profile
- crops the model profile around the Stage 1 sidewalk edge by default
- bins into 25 fixed-width bins when the sidewalk-edge buffer is enabled
- interpolates internal gaps
- smooths only through `last_valid_bin`
- zero-pads the trailing region
- writes per-bin channels and scalar engineered features into one row per station-side

Current per-bin channels:

- `z_000 ... z_024`
- `gap_000 ... gap_024`
- `pad_000 ... pad_024`

Current scalar engineered features:

- `feat_kink_dz`
- `feat_kink_slope`

Current edge metadata:

- `lane_edge_v`
- `side_edge_v`
- `interface_center_v`
- sidewalk-edge buffer audit columns such as `profile_mode`, `buffer_applied`, and `sidewalk_edge_center_bin`

Other important row metadata:

- `label`
- `class_int`
- `s_m`
- `side`
- `source_file`
- `n_valid_bins`
- `last_valid_bin`
- `n_sidewalk_pts`
- `n_lane_pts`

Important design rule:

- the metrics CSV from Stage 1 is merged by rounded `s_m` plus `side`
- scalar engineered features live in the same final row as the profile channels
- this keeps later train/validation splits automatically aligned
- profiles are currently binned from a sidewalk-edge-centered buffer by default; set `USE_SIDEWALK_EDGE_BUFFER = False` to use the full 192-bin merged lane/sidewalk profile

## Stage 3 - PyTorch Training

`pytorch_ramp_cnn_training.ipynb` trains a hybrid CNN + feature-vector classifier from one or more `cnn_dataset*.csv` files.

It currently does the following:

- loads dataset CSVs from `training/train/` and optionally `training/val/`
- supports two split modes:
  - `internal_train_split`
  - `folder_holdout`
- applies optional per-class caps to training data
- optionally augments only training `RAMP` rows by applying minute perturbations to valid `z_*`
- builds a profile tensor branch from the 25-bin `z/gap/pad` sidewalk-edge-buffer channels
- builds a scalar feature branch from `feat_kink_dz` and `feat_kink_slope`
- trains a PyTorch hybrid model
- selects best checkpoint by validation macro F1

Current model structure:

- CNN branch on profile channels `(3, 25)`
- small MLP branch on scalar feature vector `(2,)`
- concatenated classifier head

Current training behavior:

- class-weighted cross-entropy by default
- optional focal loss
- optional ramp-only augmentation on train only
- validation macro F1 is tracked every epoch
- best epoch is chosen by highest validation macro F1

Main saved outputs:

- `ramp_cnn_pytorch.pt`
- `ramp_cnn_pytorch_meta.json`

## Stage 4 - PyTorch Prediction

`pytorch_ramp_cnn_prediction.ipynb` applies the trained model to Stage 1 sidewalk/lane CSV exports.

It currently does the following:

- loads one Stage 1 `*_sidewalk.csv`
- loads one Stage 1 `*_lane.csv`
- rebuilds the same 25-bin sidewalk-edge-buffer `z/gap/pad` profile channels used for training
- recomputes `feat_kink_dz` and `feat_kink_slope` using helper functions refactored from Stage 1
- skips CNN inference for station-sides without valid kink features, such as `NO_SIDEWALK` and `SEPARATED_NO_RAMP`
- loads `ramp_cnn_pytorch.pt` and `ramp_cnn_pytorch_meta.json`
- normalizes prediction inputs using saved training statistics
- predicts CNN labels for eligible station-sides
- joins predictions back onto every sidewalk point
- writes prediction outputs into a `predictions/` folder beside the input CSVs

Main outputs from this stage:

- `<base>_cnn_predictions.csv`
- `<base>_sidewalk_cnn_pred.csv`
- `<base>_sidewalk_cnn_pred_CC.asc`
- `<base>_sidewalk_cnn_pred_CC.las`

Important design rules:

- the original Stage 1 `class` / `label` columns are preserved
- prediction columns are added separately as `pred_class`, `pred_label`, `pred_confidence`, and probabilities
- CloudCompare-style prediction outputs use `pred_R`, `pred_G`, and `pred_B` so the visualization reflects model predictions
- all new prediction files go into a local `predictions/` folder
- prediction profile construction in `ramp_prediction_helpers.py` must match the dataset profile construction used for the loaded trained model

Shared prediction helpers live in:

- `ramp_prediction_helpers.py`

## How The Stages Connect

The intended data flow is:

1. Run `updated_ditch_latest_version.ipynb`
2. If needed, manually correct station-side labels and create `*_sidewalk_remapped.csv`
3. Run `cnn_dataset_from_remapped_and_lane_csv.ipynb`
4. Copy resulting `cnn_dataset*.csv` files into the training folder structure
5. Run `pytorch_ramp_cnn_training.ipynb`
6. Run `pytorch_ramp_cnn_prediction.ipynb` on Stage 1 `*_sidewalk.csv` and `*_lane.csv` exports

Practical mapping between stages:

- Stage 1 produces point-level exports and station-side scalar metrics
- Stage 2 converts those into sample-level ML rows
- Stage 3 consumes those sample rows for model training
- Stage 4 rebuilds equivalent sample rows from Stage 1 exports, applies the trained model, and writes sidewalk prediction products

## Recommended Folder Intent

The current training notebook expects:

- `training/train/` for files used as the train pool
- `training/val/` for files used as held-out validation when `folder_holdout` is selected

Use:

- `internal_train_split` when you want to split only the contents of `training/train/`
- `folder_holdout` when you already manually decided which dataset files belong in train vs val folders

## Detailed Notebook Docs

For notebook-specific details, use:

- `docs/UPDATED_DITCH_LATEST_VERSION_DOC.md`
- `docs/CNN_DATASET_FROM_REMAPPED_AND_LANE_CSV_DOC.md`
- `docs/PYTORCH_RAMP_CNN_TRAINING_DOC.md`
- `docs/PYTORCH_RAMP_CNN_PREDICTION_DOC.md`
