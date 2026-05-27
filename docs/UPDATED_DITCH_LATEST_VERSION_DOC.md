# `updated_ditch_latest_version.ipynb`

## Purpose

This notebook is the source-of-truth rule-based corridor-processing notebook.

It:

- loads trajectory, sidewalk LAS, and lane LAS
- builds station geometry and optional spline evaluation
- optionally refines trajectory by lane-centre alignment
- extracts lane and sidewalk cross-sections along the corridor
- classifies each station-side into ramp / curb / ditch / related classes
- exports trajectory, sidewalk, and lane point products
- exports structured station-side metrics for downstream ML use
- optionally supports manual class correction and sidewalk remapping

## Main Cell Flow

Recommended run order:

1. Cell 1
2. Cell 2
3. Cell 2.5
4. Cell 2.6
5. Cell 3
6. Cell 4
7. Cell 4b
8. Cell 4d
9. Cell 4e
10. Cell 4c
11. Cell 5
12. Cell 5.5
13. Cell 6
14. Cell 7
15. Cell 8

Notes:

- Cell 2.6 is optional refinement debugging.
- Cells 4b, 4c, 4d, and 4e are optional debug/reporting helpers.
- Cells 7 and 8 are optional post-processing helpers used after main exports exist.

## What Each Main Stage Does

### Cell 1

Loads trajectory and LAS files, trims endpoint spikes if enabled, builds station ruler `s`, optional spline geometry, voxel-downsamples point clouds, builds KD-trees, and plots a trajectory heading sanity check.

### Cell 2

Extracts one local cross-section at `S_TARGET`, converts points into the local `(v, z)` frame, orients the frame so lane sits on the positive side, splits into physical left/right masks, and plots the single-station result.

### Cell 2.5

Estimates lane-centre offsets along the corridor, smooths them, and builds corrected trajectory splines `cs_x_corr(s)` and `cs_y_corr(s)`.

### Cell 2.6

Provides detailed debugging for one refinement station, including map-view and cross-section checks for the lane-centre refinement step.

### Cell 3

Runs the full corridor loop, evaluating position and heading per station, extracting lane/sidewalk slabs, orienting them, and saving raw station slices into `results`.

### Cell 4

Runs the rule-based classifier on every station-side. It smooths and detrends profiles, detects kink bounds, computes curb and ditch metrics, resolves priority between curb and ditch, and assigns the final side class. It also performs sidewalk cluster splitting so detached sidewalk pieces can be separated from the main lane-facing side.

### Cell 4b

Shows one classified station in detail with full cross-section views, smoothed and detrended profiles, second derivative, and printed decision traces.

### Cell 4c

Exports batch debug folders for all stations, including plots and a `debug.txt` summary.

### Cell 4d

Reports per-class summary statistics for the key classifier metrics.

### Cell 4e

Reports per-class bin-count statistics from the classifier debug profiles.

### Cell 5

Exports three main products:

- trajectory point cloud
- sidewalk point cloud
- lane point cloud

These are written in both human-readable CSV form and CloudCompare ASCII form.

### Cell 5.5

Exports structured station-side metrics to:

- `<base>_station_side_metrics.csv`
- `<base>_station_side_metrics.json`

Each row is keyed by rounded `s_m` and `side`, and includes kink, curb, interface, edge, and optional ditch metrics independent of the final chosen class.

Interface edge metrics exported for downstream profile-window experiments include:

- `lane_edge_v`
- `side_edge_v`
- `interface_center_v`

### Cell 6

Converts the trajectory, sidewalk, and lane exports into LAS 1.4 files with RGB.

### Cell 7

Creates a sparse manual corrector CSV template with only `s_m`, `side`, and `class`.

### Cell 8

Applies manual station-side class corrections to the main sidewalk CSV and writes remapped sidewalk outputs.

## Main Outputs

From the main export stages, this notebook can produce:

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

## Important In-Memory Objects

- `traj_xy`: trajectory XY after optional trimming
- `s`: cumulative station values
- `cs_x`, `cs_y`: original spline geometry
- `s_offsets`, `raw_offsets`, `smooth_offsets`: refinement offset data
- `cs_x_corr`, `cs_y_corr`: corrected spline geometry
- `results`: raw per-station cross-section slices
- `classified`: final per-station classification records with per-side metadata and debug payloads
- `df_traj`: exported trajectory dataframe
- `df_sw`: exported sidewalk dataframe
- `df_lane`: exported lane dataframe
- `df_station_side_metrics`: structured station-side metrics dataframe

## Why This Notebook Matters For ML

This notebook is the source of truth for:

- point-level exports used by dataset creation
- per-station-side scalar metrics such as `kink_dz` and `kink_slope`
- per-station-side edge metrics such as `lane_edge_v` and `side_edge_v`
- manual-correction-ready remapped sidewalk exports

Downstream notebooks should treat this notebook’s exports as canonical.
