# `pytorch_ramp_cnn_training.ipynb`

## Purpose

This notebook trains a hybrid PyTorch model for ramp / curb / ditch classification from one or more `cnn_dataset*.csv` files.

It consumes the dataset rows produced by `cnn_dataset_from_remapped_and_lane_csv.ipynb` and trains a model that uses:

- a CNN branch over per-bin profile channels
- optionally, a small MLP branch over scalar engineered features

## Expected Input Data

The notebook expects `cnn_dataset*.csv` files inside:

- `training/train/`
- optionally `training/val/`

Each CSV should already contain:

### Per-bin channels

- `z_000 ... z_024`
- `gap_000 ... gap_024`
- `pad_000 ... pad_024`

### Scalar engineered features

- `feat_kink_dz`
- `feat_kink_slope`

### Metadata

- `label`
- `class_int`
- `s_m`
- `side`
- `source_file`
- `dataset_csv_path` / `dataset_csv_name` are added by the training notebook at load time

## Split Modes

The notebook currently supports two split modes.

### `internal_train_split`

- loads only files from `training/train/`
- ignores `training/val/`
- applies optional class caps to the loaded train-folder data
- then internally splits those rows into train and validation

### `folder_holdout`

- loads train files from `training/train/`
- loads validation files from `training/val/`
- no random split is performed
- class caps are applied only to training rows
- validation remains untouched

## Model Structure

### Profile branch

Uses a 1D CNN on the three 25-bin profile channels:

- `z`
- `gap`
- `pad`

Current backbone:

- `Conv1d(3 -> 64, k=7)`
- `Conv1d(64 -> 128, k=5)`
- `Conv1d(128 -> 128, k=3)`
- `BatchNorm + LeakyReLU`
- `MaxPool` after first two blocks
- channel attention after the convolution stack
- `AdaptiveAvgPool1d`
- profile embedding head `Linear(128 -> 64)`

### Feature branch

When `USE_SCALAR_FEATURES = True`, uses a small MLP on the scalar engineered feature vector:

- `feat_kink_dz`
- `feat_kink_slope`

### Final head

With scalar features enabled, the profile and feature embeddings are concatenated and fed to a final classifier head.

With `USE_SCALAR_FEATURES = False`, the model skips the scalar branch and trains a profile-only CNN. The classifier still outputs the same three classes.

The current `N_BINS` setting is `25`, matching the sidewalk-edge-buffer dataset rows.

## Normalization

### Profile normalization

Only the `z` channel is normalized using train-set mean and std.

### Feature normalization

The scalar feature vector is normalized separately using train-set mean and std when `USE_SCALAR_FEATURES = True`.

Validation uses the same train-set normalization statistics.

Manual label corrections can create rows where the rule-based scalar metrics are missing. The notebook imputes missing scalar feature values using train-set finite means before normalization. When scalar features are disabled, imputation and scalar normalization are skipped.

## Profile-Only Test

To test whether the CNN profile channels are sufficient without engineered scalar metrics, set this in Cell 1:

```python
USE_SCALAR_FEATURES = False
```

This makes `ACTIVE_FEATURE_COLS` empty, builds `(n, 0)` feature tensors, skips `feature_head`, and trains only from `z/gap/pad` profile channels.

To restore the hybrid model, set:

```python
USE_SCALAR_FEATURES = True
```

## Optional Training Features

### Per-class caps

Training rows can be capped by label using `MAX_SAMPLES_PER_LABEL`.

### Ramp augmentation

Optional minute augmentation can be applied only to training `RAMP` rows:

- tiny vertical noise
- tiny vertical offset

Only the `z_*` values are changed; scalar engineered features remain inherited from the parent sample.

### Loss choice

The notebook supports:

- weighted cross-entropy
- optional focal loss

## Model Selection and Evaluation

### Best checkpoint rule

Best epoch selection is based on highest validation macro F1, not lowest validation loss.

### Metrics tracked per epoch

- train loss
- validation loss
- train accuracy
- validation accuracy
- train macro F1
- validation macro F1

### Validation reporting

The notebook reports:

- classification report
- raw confusion matrix
- normalized confusion matrix
- training curves for loss, accuracy, and macro F1

## Debugging Helpers

The notebook also contains a misclassification inspection cell that can visualize misclassified samples for any chosen ground-truth class.

It shows:

- locally centered profile shape
- gap and valid regions
- station metadata
- raw engineered feature values
- normalized engineered feature values seen by the model

## Saved Outputs

The notebook saves:

- `ramp_cnn_pytorch.pt`
- `ramp_cnn_pytorch_meta.json`

Saved metadata includes:

- split mode
- train / val directories
- train / val dataset file lists
- channel names
- scalar feature usage flag
- active feature column names
- available feature column names
- normalization statistics for `z`
- normalization statistics for scalar features
- class mappings
- best epoch and validation metrics

## Typical Workflow

1. Generate one or more `cnn_dataset*.csv` files
2. Place them in `training/train/` and optionally `training/val/`
3. Choose `SPLIT_MODE`
4. Run the notebook
5. Inspect validation confusion matrix and macro F1
6. Save the best model and metadata
