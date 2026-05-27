# `pytorch_ramp_cnn_prediction.ipynb`

## Purpose

This notebook applies the trained PyTorch ramp classifier to Stage 1 corridor exports.

It is the prediction/inference counterpart to `pytorch_ramp_cnn_training.ipynb`. It takes already-exported sidewalk and lane CSVs from `updated_ditch_latest_version.ipynb`, rebuilds the same model input structure used during training, runs the saved CNN model, and writes prediction outputs for the sidewalk point cloud.

## Expected Inputs

The notebook expects Stage 1 CSV exports for one corridor segment:

- `*_sidewalk.csv`
- `*_lane.csv`

The sidewalk CSV must contain at least:

- `x`
- `y`
- `z`
- `s_m`
- `v_m`
- `side`
- `class`
- `label`

The lane CSV must contain at least:

- `z`
- `s_m`
- `v_m`
- `side`

The notebook also expects the trained model outputs from `pytorch_ramp_cnn_training.ipynb`:

- `training/pytorch_ramp_cnn_outputs/ramp_cnn_pytorch.pt`
- `training/pytorch_ramp_cnn_outputs/ramp_cnn_pytorch_meta.json`

## Output Folder

All prediction outputs are written to a `predictions/` folder inside the same directory as the input sidewalk CSV.

Example:

```text
11/segments/part2/predictions/
```

The notebook creates this folder automatically if it does not exist.

## Main Outputs

Current outputs:

- `<base>_cnn_predictions.csv`
- `<base>_sidewalk_cnn_pred.csv`
- `<base>_sidewalk_cnn_pred_CC.asc`
- `<base>_sidewalk_cnn_pred_CC.las`

The ASC and LAS outputs use prediction RGB columns, so CloudCompare visualization reflects the CNN model predictions rather than the original rule-based labels.

## High-Level Workflow

For each station-side sample keyed by rounded `(s_m, side)`:

1. Load Stage 1 sidewalk and lane CSVs.
2. Normalize side labels and rounded station keys.
3. Rebuild the same profile tensor structure used for training:
   - `z_000 ... z_024`
   - `gap_000 ... gap_024`
   - `pad_000 ... pad_024`
4. Apply the same sidewalk-edge-buffer crop used by the focused training dataset.
5. Recompute scalar engineered features using Stage 1 classifier logic in `ramp_prediction_helpers.py`:
   - `feat_kink_dz`
   - `feat_kink_slope`
6. Split station-sides into:
   - CNN-eligible rows with valid scalar features
   - rule-based passthrough rows such as `NO_SIDEWALK` and `SEPARATED_NO_RAMP`
7. Load the trained PyTorch model and saved normalization metadata.
8. Run CNN inference on eligible rows only.
9. Recombine CNN predictions with passthrough station-sides.
10. Save station-side predictions.
11. Join predictions back onto every sidewalk point.
12. Save sidewalk prediction CSV, CloudCompare ASC, and LAS.

## Profile Construction

Prediction mirrors the dataset creation notebook. By default it applies the sidewalk-edge buffer and bins a 25-bin profile around `abs(side_edge_v)`. The previous lane/sidewalk edge-window crop and `EDGE_BUFFER_M` logic has been removed from `ramp_prediction_helpers.py`.

When `USE_SIDEWALK_EDGE_BUFFER = True`, prediction crops around `abs(side_edge_v)`, shifts the buffer start to `v = 0`, and bins the sidewalk-centered window with `SIDEWALK_EDGE_N_BINS = 25`. If disabled, prediction falls back to the full merged profile with `FULL_PROFILE_N_BINS = 192`.

Prediction rows include buffer audit metadata:

- `profile_mode`
- `sidewalk_edge_buffer_m`
- `buffer_applied`
- `buffer_start_v`
- `buffer_end_v`
- `buffer_width_m`
- `sidewalk_edge_center_v`
- `sidewalk_edge_center_bin`

Prediction also recomputes and records edge metadata:

- `lane_edge_v`
- `side_edge_v`
- `interface_center_v`

## Prediction Classes

The CNN predicts only the three classes used during training:

- `RAMP`
- `CURB_NO_RAMP`
- `DEPRESSED_DITCH`

Rows without valid kink features are not sent to the CNN. They pass through with their original Stage 1 label/class:

- `NO_SIDEWALK`
- `SEPARATED_NO_RAMP`

Passthrough rows receive:

- `prediction_source = "passthrough_rule_based"`
- `pred_label = original_label`
- `pred_class = original_class`
- `pred_confidence = 1.0`
- CNN class probabilities set to `0.0`

CNN rows receive:

- `prediction_source = "cnn_model"`
- `pred_label`
- `pred_class`
- `pred_confidence`
- `prob_RAMP`
- `prob_CURB_NO_RAMP`
- `prob_DEPRESSED_DITCH`

## Relationship To Training

The prediction notebook intentionally mirrors the training input structure from `cnn_dataset_from_remapped_and_lane_csv.ipynb` and `pytorch_ramp_cnn_training.ipynb`.

The model input has:

- profile tensor shape `(N, 3, 25)`
- scalar feature tensor shape `(N, 2)`

The three profile channels are:

- `z`
- `gap`
- `pad`

The two scalar features are:

- `feat_kink_dz`
- `feat_kink_slope`

The notebook uses normalization statistics saved by training:

- `z_mean`
- `z_std`
- `feat_mean`
- `feat_std`

The profile construction used during prediction must match the profile construction used to create the dataset that trained the loaded model.

## Helper Module

Shared prediction logic lives in:

- `ramp_prediction_helpers.py`

This helper file contains:

- class/color constants shared with Stage 1 exports
- Stage 1 feature extraction helpers copied/refactored from `updated_ditch_latest_version.ipynb`
- Stage 2 profile-building helpers copied/refactored from `cnn_dataset_from_remapped_and_lane_csv.ipynb`
- sidewalk prediction output helpers

## Cell Overview

### Cell 1

Defines imports, input paths, model paths, and prediction output paths. Creates the `predictions/` folder.

### Cell 2

Loads and validates Stage 1 sidewalk and lane CSV inputs.

### Cell 3

Builds prediction-time model input rows with the same `z/gap/pad` channels and scalar features used during training.

### Cell 4

Splits station-sides into CNN-eligible rows and rule-based passthrough rows.

### Cell 5

Defines the PyTorch model architecture and loads the trained checkpoint plus metadata.

### Cell 6

Runs CNN inference, computes probabilities/confidence, and recombines predictions with passthrough rows.

### Cell 7

Saves station-side predictions to `<base>_cnn_predictions.csv`.

### Cell 8

Joins station-side predictions back onto every sidewalk point and adds `pred_*` columns.

### Cell 9

Saves sidewalk prediction CSV, CloudCompare ASC, and LAS using prediction RGB values.

### Cell 10

Debug viewer for one station-side profile. It plots raw lane/sidewalk points, the model-ready binned profile, gap/pad flags, prediction probabilities, and feature metadata for a configurable `DEBUG_S_M` and `DEBUG_SIDE`.

## Important Output Columns

The station-side prediction CSV includes:

- `s_m`
- `s_key`
- `side`
- `original_class`
- `original_label`
- `feature_rule_class`
- `feature_rule_label`
- `pred_class`
- `pred_label`
- `pred_confidence`
- `prob_RAMP`
- `prob_CURB_NO_RAMP`
- `prob_DEPRESSED_DITCH`
- `prediction_source`
- `feat_kink_dz`
- `feat_kink_slope`
- `profile_mode`
- `sidewalk_edge_buffer_m`
- `buffer_applied`
- `buffer_start_v`
- `buffer_end_v`
- `buffer_width_m`
- `sidewalk_edge_center_v`
- `sidewalk_edge_center_bin`
- `lane_edge_v`
- `side_edge_v`
- `interface_center_v`
- `n_valid_bins`
- `last_valid_bin`
- `n_sidewalk_pts`
- `n_lane_pts`

The sidewalk prediction CSV keeps original Stage 1 columns and adds prediction columns such as:

- `pred_class`
- `pred_label`
- `pred_confidence`
- `prob_RAMP`
- `prob_CURB_NO_RAMP`
- `prob_DEPRESSED_DITCH`
- `pred_R`
- `pred_G`
- `pred_B`
- `pred_color`

## Notes

Original Stage 1 `class`, `label`, `R`, `G`, `B`, and `color` columns are preserved in the sidewalk prediction CSV.

Prediction visualization products use `pred_R`, `pred_G`, and `pred_B` so the output resembles the model predictions.
