# `cnn_dataset_from_remapped_and_lane_csv.ipynb`

## Purpose

This notebook converts one remapped sidewalk CSV plus one exported lane CSV into one ML-ready `cnn_dataset.csv`.

It is the bridge between:

- Stage 1 point-level exports from `updated_ditch_latest_version.ipynb`
- Stage 3 PyTorch model training in `pytorch_ramp_cnn_training.ipynb`

## Inputs

Expected inputs:

- `*_sidewalk_remapped.csv`
- `*_lane.csv`
- `*_station_side_metrics.csv`

The metrics file is used to merge scalar engineered features like `kink_dz` and `kink_slope` into the final dataset rows.

Required metrics columns are:

- `s_m`
- `side`
- `kink_dz`
- `kink_slope`
- `lane_edge_v`
- `side_edge_v`

If present, `interface_center_v` is also carried through to the output metadata.

## High-Level Workflow

For each station-side sample keyed by `(s_m, side)`:

1. Load sidewalk points from the remapped sidewalk CSV
2. Load lane points from the lane CSV
3. Merge sidewalk and lane points into one 1D lateral profile
4. Keep only configured target labels
5. Bin into 192 fixed-width bins
6. Mark real bins, gap-interpolated bins, and padded bins
7. Interpolate internal gaps only
8. Smooth only through `last_valid_bin`
9. Zero-pad the trailing region
10. Merge scalar metrics from `*_station_side_metrics.csv`
11. Write one final row to `cnn_dataset.csv`

## Output Structure

Each row in `cnn_dataset.csv` represents one `(s_m, side)` sample.

### Per-bin channels

- `z_000 ... z_191`
- `gap_000 ... gap_191`
- `pad_000 ... pad_191`

Definitions:

- `z_*`: final smoothed profile values
- `gap_*`: `1` where a bin was filled by interpolation inside the valid region, else `0`
- `pad_*`: `1` where a bin is trailing zero padding after `last_valid_bin`, else `0`

### Scalar engineered features

Currently:

- `feat_kink_dz`
- `feat_kink_slope`

These are merged from `*_station_side_metrics.csv` by:

- rounded `s_m` (`s_key`)
- `side`

Both are currently converted to absolute values when loaded so opposite signs do not cancel out.

### Sidewalk-edge buffer metadata

The previous lane/sidewalk edge-window crop and `EDGE_BUFFER_M` logic has been removed. The notebook now supports an optional sidewalk-centered buffer controlled by `USE_SIDEWALK_EDGE_BUFFER`.

When the flag is disabled, `profile_mode = "full_profile"` and the full merged lane/sidewalk profile is binned. When enabled, the profile is cropped around `abs(side_edge_v)`, shifted so the start of the buffer is `v = 0`, and binned with `SIDEWALK_EDGE_N_BINS`.

The output carries buffer audit metadata:

- `profile_mode`
- `sidewalk_edge_buffer_m`
- `buffer_applied`
- `buffer_start_v`
- `buffer_end_v`
- `buffer_width_m`
- `sidewalk_edge_center_v`
- `sidewalk_edge_center_bin`

The output also carries Stage 1 edge metadata:

- `lane_edge_v`
- `side_edge_v`
- `interface_center_v`

### Other metadata columns

- `label`
- `class_int`
- `s_m`
- `side`
- `source_file`
- `n_valid_bins`
- `last_valid_bin`
- `n_sidewalk_pts`
- `n_lane_pts`

## Important Design Choices

### Target-label filtering

The current `TARGET_LABELS` setting keeps only:

- `RAMP`
- `CURB_NO_RAMP`
- `DEPRESSED_DITCH`

Station-side groups with labels outside this set are skipped. With `DROP_ZERO_VALID = True`, samples that have no valid bins after combining sidewalk and lane points are also skipped.

### Current profile settings

The current default settings are:

- `BIN_SIZE = 0.08`
- `FULL_PROFILE_N_BINS = 192`
- `SIDEWALK_EDGE_N_BINS = 25`
- `USE_SIDEWALK_EDGE_BUFFER = False`
- `N_BINS = SIDEWALK_EDGE_N_BINS if USE_SIDEWALK_EDGE_BUFFER else FULL_PROFILE_N_BINS`
- `MIN_PTS_PER_BIN = 3`
- `SG_WINDOW = 11`
- `SG_POLY = 4`
- `SIDEWALK_EDGE_BUFFER_M = 1.0`

The sidewalk-edge buffer helper is present but not yet wired into sample construction. With `USE_SIDEWALK_EDGE_BUFFER = False`, the notebook keeps the full-profile default.

When enabled in a later wiring step, the helper will crop to:

```text
abs(side_edge_v) - SIDEWALK_EDGE_BUFFER_M
through
abs(side_edge_v) + SIDEWALK_EDGE_BUFFER_M
```

Then it shifts the cropped lateral coordinate so `abs(side_edge_v) - SIDEWALK_EDGE_BUFFER_M` becomes `v = 0`.

### Combined 1D profile

Lane and sidewalk points are merged into one profile before binning. This keeps the representation simple and avoids recomputing geometry in the training notebook.

### Side-local lateral axis

The notebook uses side-local `abs(v_m)` values so both left and right samples are represented as positive lateral distance from the station anchor.

### Internal interpolation only

Only gaps between observed bins are interpolated. Bins beyond the last observed valid bin are zero-padded.

### Metrics are merged, not recomputed

The notebook does not reimplement the full kink/curb logic from the main classifier. Instead, it loads the structured station-side metrics exported from Stage 1 and merges them by station-side key.

Stage 1-exported `side_edge_v` is used for the optional sidewalk-edge buffer when `USE_SIDEWALK_EDGE_BUFFER = True`. `lane_edge_v`, `side_edge_v`, and optional `interface_center_v` are also kept as output metadata.

## Cells Overview

### Cell 1

Defines input paths, binning parameters, target labels, and output path.

### Cell 2

Loads and validates:

- remapped sidewalk CSV
- lane CSV
- station-side metrics CSV

### Cell 3

Defines helper functions for:

- row grouping
- profile binning
- interpolation and padding flags
- smoothing
- lane lookup caching

### Cell 4

Builds one final sample per `(s_m, side)` group.

### Cell 5

Assembles the final dataframe and orders columns.

### Cell 6

Writes `cnn_dataset.csv`.

### Cell 7

Prints quality checks, including:

- class distribution
- valid-bin statistics
- lane/sidewalk point-count statistics
- mean/std for kink features by class

### Cell 8

Plots a few sample profiles per class.

## Why This Notebook Matters For Training

This notebook produces the single-row-per-sample format that the PyTorch notebook consumes directly.

It keeps:

- profile channels
- gap/pad indicators
- scalar engineered features
- provenance metadata

all aligned in the same CSV row, which makes later splitting and feature loading straightforward.
