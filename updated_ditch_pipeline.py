"""Core rule-based ramp detection pipeline.

This script is the non-notebook version of the core workflow in
``updated_ditch_latest_version.ipynb``. It intentionally excludes debug
viewers, plots, reporting-only cells, and manual remapping.

Pipeline summary
----------------
1. Load a trajectory CSV and two point clouds: sidewalk LAS and lane LAS.
2. Build a station axis along the trajectory and optional trajectory splines.
3. Optionally shift the trajectory toward the lane centre to reduce GPS drift.
4. At each station, cut a thin cross-section slab from lane and sidewalk points.
5. Classify the physical RIGHT and LEFT station sides independently.
6. Export point-level CSV/ASC products and station-side metrics for ML stages.

Coordinate convention
---------------------
For every station the script builds a local coordinate frame:

``u`` is along the trajectory heading, and ``v`` is lateral. ``n_hat`` is a
90-degree clockwise rotation of the heading, so positive raw ``v`` is "right"
relative to the current trajectory direction. Some trajectories are digitized
in the opposite direction, so cross-sections are later oriented with
``near_sign`` to keep the lane-facing side consistent while preserving enough
information to project points back into world XY for export.
"""

#python updated_ditch_pipeline.py --sidewalk-las "02\segments\part2\Sidewalk.las" --lane-las "02\segments\part2\Road.las" --traj-csv "02\segments\part2\traj.csv"
#python remap_station_side_classes.py --base "C:\Users\jianjing\Desktop\Fiseha\Ramp_Detection\01\segments\part4 - Copy\traj" --apply
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from sklearn.neighbors import KDTree


CLASS_CODE = {
    "RAMP": 0,
    "CURB_NO_RAMP": 1,
    "SEPARATED_NO_RAMP": 2,
    "NO_SIDEWALK": 3,
    "DEPRESSED_DITCH": 4,
    "UNKNOWN": 5,
}

CLASS_COLOR = {
    "RAMP": (0, 210, 0),
    "CURB_NO_RAMP": (255, 140, 0),
    "SEPARATED_NO_RAMP": (0, 100, 255),
    "NO_SIDEWALK": (220, 50, 50),
    "DEPRESSED_DITCH": (160, 32, 240),
    "UNKNOWN": (180, 180, 180),
}

# The class priority is only used for station-level trajectory coloring. The
# side-level labels remain independent in the sidewalk/lane exports.
PRIORITY = {name: i for i, name in enumerate(CLASS_CODE)}
INV_MAP = {v: k for k, v in CLASS_CODE.items()}
SIDE_TO_INT = {"RIGHT": 0, "LEFT": 1}


@dataclass
class PipelineConfig:
    """Runtime configuration for one corridor segment.

    The defaults are intentionally copied from the current notebook so the
    command-line script behaves like the notebook unless a flag overrides a
    value. Paths are required by the CLI, but everything else has a notebook
    default.

    The classifier parameters are grouped here instead of hidden as globals so
    a later batch runner can construct one config per zone/segment without
    changing the core processing functions.
    """

    sidewalk_las: Path
    lane_las: Path
    traj_csv: Path
    output_dir: Path | None = None
    use_spline: bool = True
    use_traj_refinement: bool = True
    write_las: bool = False

    # Corridor extraction defaults from the notebook.
    voxel: float | None = 0.05
    step_size: float = 0.2
    heading_window: float = 35.0
    half_thick_u: float = 0.4
    v_min: float = -15.0
    v_max: float = 15.0
    query_radius: float = 35.0

    # Trajectory cleanup/refinement defaults from the notebook.
    trim_endpoint_spikes: bool = True
    traj_spike_step_mult: float = 10.0
    traj_spike_abs_step_m: float = 10.0
    refine_step: float = 1.0
    refine_min_lane_pts: int = 30
    refine_sg_win: int = 51
    refine_sg_poly: int = 3
    refine_max_offset: float = 50.0

    # Rule-based classifier defaults from the notebook.
    bin_size: float = 0.08
    min_pts: int = 3
    sg_win: int = 11
    sg_poly: int = 4
    min_sw_pts: int = 50
    unknown_min_all_pts: int = 100
    gap_max: float = 1.5
    edge_band: float = 0.35
    curb_height_thresh: float = 0.05
    curb_slope_thresh: float = 0.1
    curb_override_slope_mult: float = 2.5
    use_doc_kink_metrics: bool = True
    interface_kink_buffer: float = 0.75
    use_ditch_buffer: bool = True
    ditch_buffer: float = 3.0
    ditch_priority_over_curb: bool = False
    enable_sidewalk_cluster_split: bool = False
    sidewalk_cluster_gap_m: float = 0.75
    min_cluster_sw_pts: int = 15


@dataclass
class LoadedData:
    """Input arrays and lookup structures loaded from disk.

    ``traj_xy`` and ``s`` define the station axis. ``cs_x`` and ``cs_y`` are
    spline functions from station ``s`` to world XY. ``side_tree`` and
    ``lane_tree`` are KD-trees over XY only; the Z arrays stay parallel to the
    original point arrays and are indexed by KD-tree query results.
    """

    traj_xy: np.ndarray
    traj_df: pd.DataFrame
    xcol: str
    ycol: str
    s: np.ndarray
    cs_x: CubicSpline
    cs_y: CubicSpline
    side_xy: np.ndarray
    side_z: np.ndarray
    lane_xy: np.ndarray
    lane_z: np.ndarray
    side_tree: KDTree
    lane_tree: KDTree


@dataclass
class RefinedTrajectory:
    """Optional corrected trajectory splines from lane-centre refinement.

    The corrected splines are parameterized by the original station values in
    ``s_offsets``. This matters because the corridor loop still samples by the
    original station axis; recomputing arc length after shifting the line would
    desynchronize station values from the exported point products.
    """

    enabled: bool
    s_offsets: np.ndarray | None = None
    raw_offsets: np.ndarray | None = None
    smooth_offsets: np.ndarray | None = None
    cs_x_corr: CubicSpline | None = None
    cs_y_corr: CubicSpline | None = None


@dataclass
class PipelineOutputs:
    """Paths written by the core pipeline.

    LAS paths are optional because LAS writing can be slow and is controlled by
    ``--write-las``. CSV/ASC outputs are always written because they are the
    Stage 1 products consumed by the later dataset and prediction notebooks.
    """

    traj_csv: Path
    sidewalk_csv: Path
    lane_csv: Path
    traj_asc: Path
    sidewalk_asc: Path
    lane_asc: Path
    metrics_csv: Path
    metrics_json: Path
    run_params_txt: Path
    traj_las: Path | None = None
    sidewalk_las: Path | None = None
    lane_las: Path | None = None


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Return a CSS-style hex color for RGB values in 0-255 range."""

    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def dominant_class(class_right: str, class_left: str) -> str:
    """Pick the station-level dominant class by the notebook priority order."""

    if PRIORITY.get(class_right, 99) <= PRIORITY.get(class_left, 99):
        return class_right
    return class_left


def safe_float(value, default=np.nan) -> float:
    """Convert scalar metadata to a finite float, falling back for nested objects."""

    try:
        if isinstance(value, dict) or value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def trim_trajectory_endpoint_spikes(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, int, int, float, float]:
    """Drop obvious start/end trajectory points that create huge spline jumps."""

    if not config.trim_endpoint_spikes or len(df) < 3:
        return df.reset_index(drop=True), 0, 0, np.nan, np.nan

    xy = df[[xcol, ycol]].to_numpy(float)
    dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid = dists[np.isfinite(dists) & (dists > 1e-9)]
    if len(valid) == 0:
        return df.reset_index(drop=True), 0, 0, np.nan, np.nan

    typical_step = float(np.median(valid))
    spike_thresh = float(max(config.traj_spike_abs_step_m, config.traj_spike_step_mult * typical_step))

    start_trim = 0
    while (len(df) - start_trim) >= 3:
        xy_cur = df[[xcol, ycol]].to_numpy(float)[start_trim:]
        d0 = float(np.linalg.norm(xy_cur[1] - xy_cur[0]))
        d1 = float(np.linalg.norm(xy_cur[2] - xy_cur[1]))
        if d0 > spike_thresh and d0 > 3.0 * max(d1, typical_step):
            start_trim += 1
        else:
            break

    end_trim = 0
    while (len(df) - start_trim - end_trim) >= 3:
        xy_cur = df[[xcol, ycol]].to_numpy(float)[start_trim : len(df) - end_trim]
        dn = float(np.linalg.norm(xy_cur[-1] - xy_cur[-2]))
        dprev = float(np.linalg.norm(xy_cur[-2] - xy_cur[-3]))
        if dn > spike_thresh and dn > 3.0 * max(dprev, typical_step):
            end_trim += 1
        else:
            break

    if start_trim == 0 and end_trim == 0:
        return df.reset_index(drop=True), 0, 0, typical_step, spike_thresh

    stop = len(df) - end_trim if end_trim > 0 else len(df)
    return df.iloc[start_trim:stop].reset_index(drop=True), int(start_trim), int(end_trim), typical_step, spike_thresh


def load_trajectory(csv_path: Path, config: PipelineConfig) -> tuple[np.ndarray, str, str, pd.DataFrame]:
    """Load a trajectory CSV and auto-detect its horizontal coordinate columns."""

    df = pd.read_csv(csv_path)
    cand_x = ["utm_e", "easting", "Easting", "east", "x", "X", "pos_x", "px", "lon", "Lon"]
    cand_y = ["utm_n", "northing", "Northing", "north", "y", "Y", "pos_y", "py", "lat", "Lat"]
    xcol = next((c for c in cand_x if c in df.columns), None)
    ycol = next((c for c in cand_y if c in df.columns), None)
    if xcol is None or ycol is None:
        raise ValueError(f"Could not auto-detect X/Y columns. Available columns: {list(df.columns)}")

    df, start_trim, end_trim, typical_step, spike_thresh = trim_trajectory_endpoint_spikes(df, xcol, ycol, config)
    print(f"Trajectory columns: X={xcol!r}, Y={ycol!r}")
    if start_trim or end_trim:
        print(f"Trajectory cleanup: trimmed start={start_trim}, end={end_trim}; threshold={spike_thresh:.3f} m")
    else:
        print("Trajectory cleanup: no endpoint spike rows trimmed")
    return df[[xcol, ycol]].to_numpy(float), xcol, ycol, df


def build_station_ruler(traj_xy: np.ndarray) -> np.ndarray:
    """Build cumulative station values from ordered trajectory XY points."""

    diffs = np.diff(traj_xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(traj_xy), dtype=float)
    s[1:] = np.cumsum(dists)
    return s


def build_trajectory_splines(s: np.ndarray, traj_xy: np.ndarray, min_ds: float = 1e-9) -> tuple[CubicSpline, CubicSpline, int]:
    """Build cubic splines on the strictly increasing station subset."""

    if len(s) != len(traj_xy):
        raise ValueError("s and traj_xy must have the same length")
    keep = np.ones(len(s), dtype=bool)
    if len(s) >= 2:
        keep[1:] = np.diff(s) > float(min_ds)
    s_spline = s[keep]
    xy_spline = traj_xy[keep]
    if len(s_spline) < 2:
        raise ValueError("Need at least two strictly increasing trajectory stations for spline interpolation.")
    return CubicSpline(s_spline, xy_spline[:, 0]), CubicSpline(s_spline, xy_spline[:, 1]), int(len(s) - len(s_spline))


def compute_heading(
    traj_xy: np.ndarray,
    s: np.ndarray,
    idx: int,
    window_m: float,
    s_target: float | None = None,
    cs_x: CubicSpline | None = None,
    cs_y: CubicSpline | None = None,
) -> np.ndarray | None:
    """Return a unit heading vector from spline derivatives or raw samples."""

    if s_target is None:
        s_target = float(s[idx])

    if cs_x is not None and cs_y is not None:
        deriv = np.array([float(cs_x(s_target, 1)), float(cs_y(s_target, 1))])
        length = np.linalg.norm(deriv)
        if length >= 1e-12:
            return deriv / length
        s0 = max(float(s[0]), float(s_target - window_m))
        s1 = min(float(s[-1]), float(s_target + window_m))
        if (s1 - s0) < 1e-9:
            return None
        mean_dir = np.array([float(cs_x(s1) - cs_x(s0)), float(cs_y(s1) - cs_y(s0))])
        length = np.linalg.norm(mean_dir)
        return None if length < 1e-12 else mean_dir / length

    s0 = s[idx] - window_m
    s1 = s[idx] + window_m
    i0 = max(0, int(np.searchsorted(s, s0, side="left")))
    i1 = min(len(s) - 1, int(np.searchsorted(s, s1, side="right")) - 1)
    if i1 - i0 < 2:
        return None
    steps = traj_xy[i0 + 1 : i1 + 1] - traj_xy[i0:i1]
    norms = np.linalg.norm(steps, axis=1)
    steps = steps[norms > 1e-9]
    if len(steps) == 0:
        return None
    mean_dir = np.mean(steps, axis=0)
    length = np.linalg.norm(mean_dir)
    return None if length < 1e-12 else mean_dir / length


def read_las_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read XYZ arrays from a LAS/LAZ file using laspy."""

    las = laspy.read(path)
    return np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)


def voxel_downsample(x: np.ndarray, y: np.ndarray, z: np.ndarray, voxel: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample points by keeping the first point in each voxel grid cell."""

    if voxel is None or voxel <= 0:
        return x, y, z
    pts = np.column_stack([x, y, z])
    keys = np.floor(pts / float(voxel)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    pts = pts[idx]
    return pts[:, 0], pts[:, 1], pts[:, 2]


def load_inputs(config: PipelineConfig) -> LoadedData:
    """Load trajectory/LAS inputs and build station splines plus KD-trees.

    This function performs all expensive input setup once. Later corridor
    processing repeatedly asks the KD-trees for nearby points, so we keep the
    XY arrays and Z arrays in memory rather than reading from disk per station.
    """

    traj_xy, xcol, ycol, traj_df = load_trajectory(config.traj_csv, config)
    s = build_station_ruler(traj_xy)
    cs_x, cs_y, dropped = build_trajectory_splines(s, traj_xy)
    print(f"Trajectory: {len(traj_xy):,} points, length={s[-1]:.1f} m, spline_dropped={dropped}")

    print("Loading LAS point clouds...")
    sx, sy, sz = read_las_xyz(config.sidewalk_las)
    lx, ly, lz = read_las_xyz(config.lane_las)
    print(f"  Sidewalk raw: {len(sx):,} pts | Lane raw: {len(lx):,} pts")
    sx, sy, sz = voxel_downsample(sx, sy, sz, config.voxel)
    lx, ly, lz = voxel_downsample(lx, ly, lz, config.voxel)
    print(f"  Sidewalk voxel: {len(sx):,} pts | Lane voxel: {len(lx):,} pts")

    # KDTree only indexes horizontal position. Elevation is carried in the
    # parallel Z arrays and recovered with the point indices returned by queries.
    side_xy = np.column_stack([sx, sy])
    lane_xy = np.column_stack([lx, ly])
    return LoadedData(
        traj_xy=traj_xy,
        traj_df=traj_df,
        xcol=xcol,
        ycol=ycol,
        s=s,
        cs_x=cs_x,
        cs_y=cs_y,
        side_xy=side_xy,
        side_z=sz,
        lane_xy=lane_xy,
        lane_z=lz,
        side_tree=KDTree(side_xy),
        lane_tree=KDTree(lane_xy),
    )


def extract_cross_section(
    xy_pts: np.ndarray,
    z_pts: np.ndarray,
    tree: KDTree,
    p: np.ndarray,
    t_hat: np.ndarray,
    half_u: float,
    v_min: float,
    v_max: float,
    query_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract points in a local cross-section slab perpendicular to the road.

    The KD-tree radius query is a fast coarse filter. After that, points are
    projected into the station-local frame:

    - ``u`` measures along-road distance from the station anchor.
    - ``v`` measures lateral distance from the station anchor.

    Only points with ``abs(u) <= half_u`` form the thin cross-section slab.
    ``v_min``/``v_max`` keep the lateral profile bounded so a station cannot
    accidentally collect unrelated points far away from the corridor.
    """

    # Coarse spatial lookup: find nearby points in world XY.
    ids = tree.query_radius(p.reshape(1, -1), r=query_r)[0]
    if len(ids) == 0:
        return np.array([]), np.array([])

    # Build local coordinates from the station anchor p. The normal is a
    # clockwise rotation of the tangent, matching the notebook convention.
    rel = xy_pts[ids] - p
    u = rel @ t_hat
    n_hat = np.array([t_hat[1], -t_hat[0]])
    v = rel @ n_hat

    # Final precise slab filter after the coarse radius query.
    mask = (np.abs(u) <= half_u) & (v >= v_min) & (v <= v_max)
    return v[mask], z_pts[ids][mask]


def orient_cross_section(v_lane_raw: np.ndarray, v_side_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Flip lateral coordinates so the lane sits on the positive-v side.

    Some input trajectories are digitized opposite the intended driving
    direction. If the median lane points are on negative raw ``v``, the profile
    is flipped before classification. ``near_sign`` records that flip so export
    code can reconstruct the original world XY locations.
    """

    sign = 1.0 if len(v_lane_raw) == 0 or np.median(v_lane_raw) >= 0 else -1.0
    return v_lane_raw * sign, v_side_raw * sign, sign


def get_physical_side_masks(v_lane: np.ndarray, v_side: np.ndarray, near_sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lane/sidewalk masks for physical RIGHT and LEFT station sides.

    ``orient_cross_section`` may flip signs to stabilize classification. This
    function converts those oriented signs back into physical RIGHT/LEFT masks
    so output labels are not swapped when a trajectory direction changes.
    """

    if near_sign >= 0:
        return v_lane >= 0, v_side >= 0, v_lane <= 0, v_side <= 0
    return v_lane <= 0, v_side <= 0, v_lane >= 0, v_side >= 0


def refine_trajectory(data: LoadedData, config: PipelineConfig) -> RefinedTrajectory:
    """Estimate lane-centre offsets and build corrected trajectory splines.

    The raw trajectory may be offset from the road centre. This pass samples
    the lane point cloud at coarse station spacing, estimates the lane centre
    as the midpoint between the 5th and 95th lateral percentiles, smooths that
    offset curve, and shifts the trajectory by the smoothed lateral offsets.

    The corrected splines are only used within the station range where offsets
    were actually estimated. Outside that range the corridor loop falls back to
    the original spline to avoid uncontrolled extrapolation.
    """

    if not config.use_traj_refinement:
        return RefinedTrajectory(enabled=False)

    print("Refining trajectory by lane-centre alignment...")
    refine_stations = np.arange(0.0, data.s[-1], config.refine_step)
    s_offsets: list[float] = []
    raw_offsets: list[float] = []

    # Pass 1: sample many stations and estimate how far the trajectory anchor
    # is from the local lane centre in the lateral v direction.
    for s_target in refine_stations:
        idx = int(np.argmin(np.abs(data.s - s_target)))
        if config.use_spline:
            p_ref = np.array([float(data.cs_x(s_target)), float(data.cs_y(s_target))])
            heading_cs_x, heading_cs_y = data.cs_x, data.cs_y
        else:
            p_ref = data.traj_xy[idx]
            heading_cs_x = heading_cs_y = None

        t_ref = compute_heading(data.traj_xy, data.s, idx, config.heading_window, float(s_target), heading_cs_x, heading_cs_y)
        if t_ref is None:
            continue
        v_lane_ref, _ = extract_cross_section(
            data.lane_xy,
            data.lane_z,
            data.lane_tree,
            p_ref,
            t_ref,
            config.half_thick_u,
            config.v_min,
            config.v_max,
            config.query_radius,
        )
        if len(v_lane_ref) < config.refine_min_lane_pts:
            continue

        # Percentiles are more stable than min/max because lane slabs often
        # contain sparse outliers at road edges or neighbouring surfaces.
        lane_centre_v = 0.5 * (float(np.percentile(v_lane_ref, 5)) + float(np.percentile(v_lane_ref, 95)))
        if abs(lane_centre_v) > config.refine_max_offset:
            continue
        s_offsets.append(float(s_target))
        raw_offsets.append(lane_centre_v)

    s_offsets_arr = np.asarray(s_offsets, dtype=float)
    raw_offsets_arr = np.asarray(raw_offsets, dtype=float)
    if len(s_offsets_arr) < 2:
        print("  Refinement skipped: fewer than 2 valid offset stations")
        return RefinedTrajectory(enabled=False)

    # Pass 2: smooth the offset curve so the corrected trajectory is stable
    # station-to-station instead of following noisy local point distributions.
    sg_win = config.refine_sg_win
    if sg_win >= len(raw_offsets_arr):
        sg_win = len(raw_offsets_arr) - 1 if (len(raw_offsets_arr) - 1) % 2 == 1 else len(raw_offsets_arr) - 2
    if sg_win < 5:
        smooth_offsets = raw_offsets_arr
    else:
        smooth_offsets = savgol_filter(raw_offsets_arr, window_length=sg_win, polyorder=config.refine_sg_poly)

    x_corr = np.zeros(len(s_offsets_arr), dtype=float)
    y_corr = np.zeros(len(s_offsets_arr), dtype=float)

    # Pass 3: shift each sampled trajectory point by its smoothed lateral
    # offset. The shift direction is the local right-normal n_hat.
    for i, (s_t, offset) in enumerate(zip(s_offsets_arr, smooth_offsets)):
        if config.use_spline:
            px = float(data.cs_x(s_t))
            py = float(data.cs_y(s_t))
            heading_cs_x, heading_cs_y = data.cs_x, data.cs_y
        else:
            idx_i = int(np.argmin(np.abs(data.s - s_t)))
            px, py = data.traj_xy[idx_i]
            heading_cs_x = heading_cs_y = None
        idx_i = int(np.argmin(np.abs(data.s - s_t)))
        t_i = compute_heading(data.traj_xy, data.s, idx_i, config.heading_window, float(s_t), heading_cs_x, heading_cs_y)
        if t_i is None:
            x_corr[i] = px
            y_corr[i] = py
            continue
        n_i = np.array([t_i[1], -t_i[0]])
        x_corr[i] = px + offset * n_i[0]
        y_corr[i] = py + offset * n_i[1]

    # Keep the corrected spline parameterized by original station values. Do
    # not recompute station from traj_corr for the spline domain.
    traj_corr = np.column_stack([x_corr, y_corr])
    cs_x_corr, cs_y_corr, dropped = build_trajectory_splines(s_offsets_arr, traj_corr)
    print(f"  Offset stations: {len(s_offsets_arr):,}, corrected range={s_offsets_arr[0]:.1f}-{s_offsets_arr[-1]:.1f} m, dropped={dropped}")
    return RefinedTrajectory(True, s_offsets_arr, raw_offsets_arr, smooth_offsets, cs_x_corr, cs_y_corr)


def build_binned_profile(v: np.ndarray, z: np.ndarray, bin_size: float, min_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate raw lateral/elevation scatter into median-elevation bins."""

    v = np.asarray(v)
    z = np.asarray(z)
    ok = np.isfinite(v) & np.isfinite(z)
    v = v[ok]
    z = z[ok]
    if len(v) == 0:
        return np.array([]), np.array([])
    edges = np.arange(float(np.min(v)), float(np.max(v)) + bin_size, bin_size)
    if len(edges) < 2:
        edges = np.array([float(np.min(v)), float(np.max(v)) + bin_size])
    idx = np.digitize(v, edges) - 1
    v_ctr = (edges[:-1] + edges[1:]) * 0.5
    z_med = np.full(len(v_ctr), np.nan)
    for b in range(len(v_ctr)):
        sel = idx == b
        if np.sum(sel) >= min_pts:
            z_med[b] = np.median(z[sel])
    return v_ctr, z_med


def smooth_savgol(z: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Smooth a profile with Savitzky-Golay, shrinking the window if needed."""

    z = np.asarray(z)
    win = int(win)
    if len(z) < 7:
        return z.copy()
    if win % 2 == 0:
        win += 1
    if win >= len(z):
        win = len(z) - 1 if (len(z) - 1) % 2 == 1 else len(z) - 2
    if win < 7:
        return z.copy()
    return savgol_filter(z, window_length=win, polyorder=poly, mode="interp")


def detrend_profile(v: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Remove a linear trend from a smoothed profile."""

    v = np.asarray(v)
    z = np.asarray(z)
    ok = np.isfinite(v) & np.isfinite(z)
    if np.sum(ok) < 2:
        return z.copy(), (0.0, 0.0)
    a, b = np.polyfit(v[ok], z[ok], 1)
    return z - (a * v + b), (float(a), float(b))


def build_full_profile(v_all: np.ndarray, z_all: np.ndarray, config: PipelineConfig) -> dict | None:
    """Build the 1D lateral elevation profile used by the classifier.

    The classifier works on a single merged lane+sidewalk profile for one
    physical side. Raw point scatter is converted into a stable profile by:

    1. Sorting points by lateral coordinate ``v``.
    2. Binning ``z`` values by median elevation.
    3. Interpolating internal sparse bins.
    4. Smoothing with Savitzky-Golay.
    5. Removing a linear trend so ditch/valley curvature is easier to detect.

    ``None`` means the profile is too sparse to classify reliably.
    """

    v_all = np.asarray(v_all)
    z_all = np.asarray(z_all)
    ok = np.isfinite(v_all) & np.isfinite(z_all)
    v_all = v_all[ok]
    z_all = z_all[ok]
    if len(v_all) < 20:
        return None

    # Sort by lateral position so interpolation and gradient calculations move
    # monotonically across the cross-section.
    order = np.argsort(v_all)
    v_all = v_all[order]
    z_all = z_all[order]
    v_ctr, z_med = build_binned_profile(v_all, z_all, config.bin_size, config.min_pts)
    if len(v_ctr) == 0:
        return None
    ok2 = np.isfinite(z_med)
    if np.sum(ok2) < 12:
        return None

    # Fill only after binning. This gives continuous arrays for smoothing and
    # derivatives while still allowing sparse profiles to be rejected above.
    z_fill = z_med.copy()
    z_fill[~ok2] = np.interp(v_ctr[~ok2], v_ctr[ok2], z_fill[ok2])

    # Ditch detection uses detrended derivatives; kink metrics use the smoothed
    # profile when config.use_doc_kink_metrics is enabled.
    z_s = smooth_savgol(z_fill, config.sg_win, config.sg_poly)
    z_dt, trend = detrend_profile(v_ctr, z_s)
    d1 = np.gradient(z_dt, config.bin_size)
    d2 = np.gradient(d1, config.bin_size)
    return {"v_ctr": v_ctr, "z_med": z_med, "z_fill": z_fill, "z_s": z_s, "z_dt": z_dt, "d1": d1, "d2": d2, "trend_coef": trend}


def pick_two_kinks(v_ctr: np.ndarray, z_s: np.ndarray, bin_size: float, v_lo: float | None, v_hi: float | None, min_bins: int = 5):
    """Pick transition kink bounds using second-derivative extrema.

    When edge metrics are available, the search is constrained to a window
    around the lane/sidewalk interface. That keeps kink-derived dz/slope focused
    on the actual transition instead of the far ends of a wide profile. If the
    window is missing or sparse, the function falls back to profile endpoints.
    """

    d1 = np.gradient(z_s, bin_size)
    d2 = np.gradient(d1, bin_size)
    use_window = False
    note = "ok"
    if v_lo is not None and v_hi is not None and np.isfinite(v_lo) and np.isfinite(v_hi):
        lo = float(min(v_lo, v_hi))
        hi = float(max(v_lo, v_hi))
        idx = np.where((v_ctr >= lo) & (v_ctr <= hi))[0]
        if len(idx) >= int(min_bins):
            i_p = int(idx[np.argmax(d2[idx])])
            i_n = int(idx[np.argmin(d2[idx])])
            if i_p != i_n:
                use_window = True
            else:
                note = "interface_window_degenerate_extrema"
        else:
            note = "interface_window_too_sparse"
    else:
        note = "interface_window_unavailable"
    if not use_window:
        i_p = 0
        i_n = len(v_ctr) - 1
        mode = "profile_ends_fallback"
    else:
        mode = "interface_window"
    return float(v_ctr[i_p]), float(v_ctr[i_n]), i_p, i_n, d2, mode, note


def estimate_side_interface(v_lane_side: np.ndarray, z_lane_side: np.ndarray, v_side_side: np.ndarray, z_side_side: np.ndarray, edge_band: float) -> dict | None:
    """Measure the lane/sidewalk facing-edge gap and elevation step.

    The facing edges are estimated with lateral percentiles rather than exact
    extrema. On the right side the outer lane edge is p95 and the inner sidewalk
    edge is p05; on the left side those percentiles are reversed. Median Z in a
    narrow band around each edge gives the interface height difference.
    """

    v_lane_side = np.asarray(v_lane_side)
    z_lane_side = np.asarray(z_lane_side)
    v_side_side = np.asarray(v_side_side)
    z_side_side = np.asarray(z_side_side)
    ok_l = np.isfinite(v_lane_side) & np.isfinite(z_lane_side)
    ok_s = np.isfinite(v_side_side) & np.isfinite(z_side_side)
    v_lane_side = v_lane_side[ok_l]
    z_lane_side = z_lane_side[ok_l]
    v_side_side = v_side_side[ok_s]
    z_side_side = z_side_side[ok_s]
    if len(v_lane_side) < 10 or len(v_side_side) < 10:
        return None

    # Percentile edge selection mirrors the separation-gap gate in
    # classify_side, so all interface metrics use the same geometry concept.
    on_right = np.median(v_lane_side) >= 0
    if on_right:
        lane_edge_v = float(np.percentile(v_lane_side, 95))
        side_edge_v = float(np.percentile(v_side_side, 5))
    else:
        lane_edge_v = float(np.percentile(v_lane_side, 5))
        side_edge_v = float(np.percentile(v_side_side, 95))
    lane_band = np.abs(v_lane_side - lane_edge_v) <= edge_band
    side_band = np.abs(v_side_side - side_edge_v) <= edge_band
    if np.sum(lane_band) < 3 or np.sum(side_band) < 3:
        return None
    z_lane_edge = float(np.median(z_lane_side[lane_band]))
    z_side_edge = float(np.median(z_side_side[side_band]))
    gap_m = float(abs(side_edge_v - lane_edge_v))
    dz_edge = float(z_side_edge - z_lane_edge)
    return {
        "lane_edge_v": lane_edge_v,
        "side_edge_v": side_edge_v,
        "z_lane_edge": z_lane_edge,
        "z_side_edge": z_side_edge,
        "gap_m": gap_m,
        "dz_edge": dz_edge,
        "edge_slope": abs(dz_edge) / max(gap_m, 1e-6),
    }


def classify_curb_from_interface(interface: dict | None, config: PipelineConfig) -> dict:
    """Classify whether interface edge metrics are curb-like."""

    if interface is None:
        return {"is_curb": False, "curb_reason": "no_interface"}
    dz = abs(interface["dz_edge"])
    slope = interface["edge_slope"]
    is_curb = dz >= config.curb_height_thresh and slope >= config.curb_slope_thresh
    return {"is_curb": bool(is_curb), "curb_reason": "edge_step_detected" if is_curb else "no_clear_edge_step", **interface}


def summarize_profile_geometry(v_ctr: np.ndarray, z_dt: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> dict:
    """Compute valley geometry used by the depressed-ditch classifier.

    Ditches are detected as concave-up depressions in the detrended profile.
    The returned metrics describe the valley minimum, approximate shoulder
    heights, depth, widths at 50 percent and 30 percent depth, and local
    curvature near the minimum.
    """

    i_min = int(np.argmin(z_dt))
    i_max = int(np.argmax(z_dt))
    zmin = float(z_dt[i_min])
    zmax = float(z_dt[i_max])
    n_end = max(3, len(z_dt) // 5)
    z_l = float(np.median(z_dt[:n_end]))
    z_r = float(np.median(z_dt[-n_end:]))

    # Valley depth is limited by the lower of the two shoulders. This prevents
    # a one-sided slope from looking like a deep symmetric ditch.
    left_depth = z_l - zmin
    right_depth = z_r - zmin
    valley_depth = min(left_depth, right_depth)
    if 1 <= i_min <= len(d1) - 2:
        left_s = float(np.median(d1[:i_min]))
        right_s = float(np.median(d1[i_min + 1 :]))
        sign_change = left_s < 0 and right_s > 0
    else:
        left_s = np.nan
        right_s = np.nan
        sign_change = False
    i0 = max(0, i_min - 1)
    i1 = min(len(d2), i_min + 2)
    valley_curv = float(np.median(d2[i0:i1]))
    if valley_depth > 0:
        # Width thresholds help distinguish a narrow depressed channel from a
        # broad terrain sag or normal road crown variation.
        idx50 = np.where(z_dt <= zmin + 0.5 * valley_depth)[0]
        idx30 = np.where(z_dt <= zmin + 0.3 * valley_depth)[0]
        valley_width_50 = float(v_ctr[idx50[-1]] - v_ctr[idx50[0]]) if len(idx50) >= 2 else np.nan
        valley_width_30 = float(v_ctr[idx30[-1]] - v_ctr[idx30[0]]) if len(idx30) >= 2 else np.nan
    else:
        valley_width_50 = np.nan
        valley_width_30 = np.nan
    total_width = float(v_ctr[-1] - v_ctr[0])
    dz_end = float(z_r - z_l)
    return {
        "i_min": i_min,
        "i_max": i_max,
        "vmin": float(v_ctr[i_min]),
        "vmax": float(v_ctr[i_max]),
        "zmin": zmin,
        "zmax": zmax,
        "zL": z_l,
        "zR": z_r,
        "left_depth": float(left_depth),
        "right_depth": float(right_depth),
        "valley_depth": float(valley_depth),
        "total_width": total_width,
        "dz_end": dz_end,
        "avg_slope": dz_end / max(total_width, 1e-9),
        "left_slope_median": left_s,
        "right_slope_median": right_s,
        "slope_sign_change_around_min": bool(sign_change),
        "valley_curv": valley_curv,
        "valley_width_50": float(valley_width_50) if np.isfinite(valley_width_50) else np.nan,
        "valley_width_30": float(valley_width_30) if np.isfinite(valley_width_30) else np.nan,
        "z_range": float(np.nanmax(z_dt) - np.nanmin(z_dt)),
    }


def classify_ditch(prof: dict | None, config: PipelineConfig, v_center: float = np.nan, min_bins: int = 5) -> dict:
    """Detect depressed ditch geometry, optionally inside the interface buffer.

    The ditch detector intentionally operates on the detrended profile, not raw
    elevations, because road crown or terrain tilt can otherwise dominate the
    absolute Z values. When ``use_ditch_buffer`` is enabled, only a window
    around the lane/sidewalk interface is evaluated so distant low ground does
    not override the curb/ramp decision near the crossing transition.
    """

    out = {
        "ditch_found": False,
        "geom_class": "UNKNOWN",
        "note": "sparse_profile",
        "ditch_eval_mode": "interface_buffer" if config.use_ditch_buffer else "full_profile",
        "ditch_window_v0": np.nan,
        "ditch_window_v1": np.nan,
        "ditch_window_center_v": float(v_center) if np.isfinite(v_center) else np.nan,
        "ditch_window_note": "buffer_requested" if config.use_ditch_buffer else "buffer_disabled",
        "ditch_window_bins": np.nan,
    }
    if prof is None:
        return out
    v_use = prof["v_ctr"]
    z_use = prof["z_dt"]
    d1_use = prof["d1"]
    d2_use = prof["d2"]
    if config.use_ditch_buffer:
        if not np.isfinite(v_center):
            out.update({"geom_class": "NOT_DITCH", "note": "ditch_buffer_center_unavailable", "ditch_window_note": "ditch_buffer_center_unavailable"})
            return out

        # Restrict ditch evaluation to an interface-centered window. This is a
        # practical guardrail against classifying remote drainage/terrain as the
        # station-side transition condition.
        v_lo = float(v_center - config.ditch_buffer)
        v_hi = float(v_center + config.ditch_buffer)
        mask = (prof["v_ctr"] >= v_lo) & (prof["v_ctr"] <= v_hi)
        out.update({"ditch_window_v0": v_lo, "ditch_window_v1": v_hi, "ditch_window_bins": int(np.count_nonzero(mask))})
        if np.count_nonzero(mask) < int(min_bins):
            out.update({"geom_class": "NOT_DITCH", "note": "ditch_buffer_too_sparse", "ditch_window_note": "ditch_buffer_too_sparse"})
            return out
        v_use = prof["v_ctr"][mask]
        z_use = prof["z_dt"][mask]
        d1_use = prof["d1"][mask]
        d2_use = prof["d2"][mask]
        out["ditch_window_note"] = "ok"
    else:
        out.update({"ditch_window_v0": float(prof["v_ctr"][0]), "ditch_window_v1": float(prof["v_ctr"][-1]), "ditch_window_bins": int(len(prof["v_ctr"]))})

    geom = summarize_profile_geometry(v_use, z_use, d1_use, d2_use)
    left_med = geom["left_slope_median"]
    right_med = geom["right_slope_median"]
    weak_sign_ok = np.isfinite(left_med) and np.isfinite(right_med) and left_med < -0.003 and right_med > -0.002
    sign_ok = geom["slope_sign_change_around_min"] or weak_sign_ok
    width50 = geom["valley_width_50"]
    width30 = geom["valley_width_30"]
    # These thresholds are the notebook's current ditch definition. They require
    # a measurable depression, concave-up curvature, and a channel-like width.
    ditch_found = bool(
        geom["valley_depth"] >= 0.04
        and sign_ok
        and geom["valley_curv"] > 0.5
        and np.isfinite(width50)
        and np.isfinite(width30)
        and 0.30 <= width50
        and width30 <= 0.6
    )
    out.update({"ditch_found": ditch_found, "geom_class": "DEPRESSED_DITCH" if ditch_found else "NOT_DITCH", "note": "ok" if ditch_found else "thresholds_not_met"})
    out.update(geom)
    return out


def build_sidewalk_clusters(v_side_side: np.ndarray, z_side_side: np.ndarray, on_right: bool, assigned_class: str, config: PipelineConfig) -> list[dict]:
    """Split a physical side into optional lateral sidewalk clusters for export.

    The side-level classifier produces one label for a whole physical side. Some
    station sides contain multiple disconnected sidewalk pieces, so this helper
    can mark meaningful detached clusters as ``SEPARATED_NO_RAMP`` while keeping
    the lane-facing cluster on the side-level label. The feature is disabled by
    default to match the current notebook defaults, but the export fields remain
    present for consistency.
    """

    v_side_side = np.asarray(v_side_side)
    z_side_side = np.asarray(z_side_side)
    if len(v_side_side) == 0:
        return []
    ok = np.isfinite(v_side_side) & np.isfinite(z_side_side)
    if not np.any(ok):
        return []
    valid_local_idx = np.flatnonzero(ok)
    v_valid = v_side_side[ok]
    order = np.argsort(v_valid)
    v_sorted = v_valid[order]
    idx_sorted = valid_local_idx[order]

    # A large lateral gap between sorted sidewalk points indicates disconnected
    # pieces on the same physical side.
    split_after = np.where(np.diff(v_sorted) > config.sidewalk_cluster_gap_m)[0]
    starts = np.concatenate(([0], split_after + 1))
    stops = np.concatenate((split_after + 1, [len(v_sorted)]))
    clusters = []
    for cluster_id, (i0, i1) in enumerate(zip(starts, stops)):
        idx_local = idx_sorted[i0:i1].astype(int)
        v_cluster = v_side_side[idx_local]
        clusters.append({
            "cluster_id": int(cluster_id),
            "point_idx_local": idx_local,
            "n_pts": int(len(idx_local)),
            "v_min": float(np.min(v_cluster)),
            "v_max": float(np.max(v_cluster)),
            "inner_edge_v": float(np.min(v_cluster)) if on_right else float(np.max(v_cluster)),
            "gap_from_previous_cluster_m": np.nan if cluster_id == 0 else float(v_sorted[i0] - v_sorted[i0 - 1]),
        })
    meaningful = [c for c in clusters if c["n_pts"] >= config.min_cluster_sw_pts]

    # The primary cluster is the meaningful cluster closest to the lane-facing
    # edge; farther meaningful clusters can optionally be marked detached.
    primary_cluster_id = (min(meaningful, key=lambda c: c["inner_edge_v"])["cluster_id"] if on_right else max(meaningful, key=lambda c: c["inner_edge_v"])["cluster_id"]) if meaningful else clusters[0]["cluster_id"]
    split_applied = bool(config.enable_sidewalk_cluster_split and len(meaningful) > 1)
    lane_rank = [c["cluster_id"] for c in (sorted(meaningful, key=lambda c: c["inner_edge_v"]) if on_right else sorted(meaningful, key=lambda c: c["inner_edge_v"], reverse=True))] if meaningful else [clusters[0]["cluster_id"]]
    lane_rank_map = {cid: rank for rank, cid in enumerate(lane_rank)}
    for cluster in clusters:
        is_meaningful = bool(cluster["n_pts"] >= config.min_cluster_sw_pts)
        is_primary = bool(cluster["cluster_id"] == primary_cluster_id)
        is_detached = bool(split_applied and is_meaningful and not is_primary)
        if is_detached:
            role = "detached"
            label = "SEPARATED_NO_RAMP"
        elif is_primary or not split_applied:
            role = "primary"
            label = assigned_class
        else:
            role = "noise"
            label = assigned_class
        cluster.update({"cluster_role": role, "cluster_label": label, "is_primary": is_primary, "is_detached": is_detached, "is_meaningful": is_meaningful, "cluster_order_from_lane": int(lane_rank_map.get(cluster["cluster_id"], -1)), "split_applied": split_applied})
    return clusters


def classify_side(v_lane: np.ndarray, z_lane: np.ndarray, v_side: np.ndarray, z_side: np.ndarray, config: PipelineConfig) -> tuple[dict, dict | None]:
    """Classify one physical station side into ramp/curb/ditch/separated/etc.

    Decision order mirrors the notebook:

    1. Not enough sidewalk points -> ``NO_SIDEWALK``.
    2. Lane/sidewalk facing-edge gap too large -> ``SEPARATED_NO_RAMP``.
    3. Too few combined points/profile bins -> ``UNKNOWN``.
    4. Build transition metrics around interface kinks.
    5. Evaluate curb and depressed-ditch candidates.
    6. Resolve curb-vs-ditch priority.
    7. If neither candidate wins, classify as ``RAMP``.

    The returned ``meta`` dict is compact and exported in point products. The
    returned ``dbg`` dict contains richer profile arrays/metrics used for
    station-side metrics export; it is not used for plotting in this script.
    """

    def empty(cls: str, gap=np.nan) -> tuple[dict, None]:
        return {"class": cls, "gap_m": gap, "dz_m": np.nan, "slope": np.nan, "transition_width_m": np.nan, "kink1_v": np.nan, "kink2_v": np.nan, "ambiguous": 0}, None

    on_right = len(v_lane) == 0 or np.median(v_lane) >= 0

    # Gate 1: without enough sidewalk points there is no transition to classify.
    if len(v_side) < config.min_sw_pts:
        return empty("NO_SIDEWALK")

    # Gate 2: if lane and sidewalk facing edges are too far apart, it is not a
    # ramp/curb interface even if both surfaces exist in the cross-section.
    gap = np.nan
    if len(v_lane) > 30 and len(v_side) > 30:
        lane_edge = np.percentile(v_lane, 95) if on_right else np.percentile(v_lane, 5)
        sw_edge = np.percentile(v_side, 5) if on_right else np.percentile(v_side, 95)
        gap = float(abs(sw_edge - lane_edge))
        if gap > config.gap_max:
            return empty("SEPARATED_NO_RAMP", gap)

    # Gate 3: the profile classifier needs enough combined support to build a
    # stable binned profile and derivatives.
    v_all = np.concatenate([v_lane, v_side])
    z_all = np.concatenate([z_lane, z_side])
    if len(v_all) < config.unknown_min_all_pts:
        return empty("UNKNOWN", gap)
    prof = build_full_profile(v_all, z_all, config)
    if prof is None:
        return empty("UNKNOWN", gap)

    v_ctr = prof["v_ctr"]
    z_s = prof["z_s"]
    z_dt = prof["z_dt"]
    interface = estimate_side_interface(v_lane, z_lane, v_side, z_side, config.edge_band)
    curb_info = classify_curb_from_interface(interface, config)
    interface_center_v = np.nan
    v_lo = v_hi = None
    if interface is not None and np.isfinite(interface.get("lane_edge_v", np.nan)) and np.isfinite(interface.get("side_edge_v", np.nan)):
        # Kink search is focused around the boundary between lane and sidewalk.
        interface_center_v = 0.5 * float(interface["lane_edge_v"] + interface["side_edge_v"])
        v_lo = interface_center_v - config.interface_kink_buffer
        v_hi = interface_center_v + config.interface_kink_buffer

    # Kink bounds define the transition width and dz/slope metrics used by the
    # current curb/ramp decision logic.
    kink1, kink2, i_k1, i_k2, d2_kink, kink_mode, kink_note = pick_two_kinks(v_ctr, z_s, config.bin_size, v_lo, v_hi)
    v0, v1 = sorted([kink1, kink2])
    width = float(v1 - v0)
    i0 = int(np.argmin(np.abs(v_ctr - v0)))
    i1 = int(np.argmin(np.abs(v_ctr - v1)))
    dz = float(z_s[i1] - z_s[i0]) if config.use_doc_kink_metrics else float(z_dt[i1] - z_dt[i0])
    dz_source = "smoothed_kink_bounds" if config.use_doc_kink_metrics else "detrended_kink_bounds"
    slope = abs(dz) / max(width, 1e-9)
    dbg = {**prof, "d2": d2_kink, "v0": v0, "v1": v1, "kink1": kink1, "kink2": kink2, "i_k1": i_k1, "i_k2": i_k2, "kink_mode": kink_mode, "kink_note": kink_note, "kink_window_v0": v_lo, "kink_window_v1": v_hi, "interface_center_v": interface_center_v, "dz_kink": dz, "slope_kink": slope, "dz_source": dz_source}
    dbg.update(curb_info)
    if interface is not None:
        dbg.update(interface)
    if width <= 0.05:
        return {"class": "UNKNOWN", "note": "no_clear_kinks", "gap_m": gap, "dz_m": np.nan, "slope": np.nan, "transition_width_m": width, "kink1_v": kink1, "kink2_v": kink2, "kink_mode": kink_mode, "kink_note": kink_note, "ambiguous": 0}, dbg

    # Curb and ditch are both evaluated before priority is resolved. This lets
    # metrics export preserve both candidates even when only one class wins.
    ditch_info = classify_ditch(prof, config, interface_center_v)
    dbg.update(ditch_info)
    curb_from_interface = bool(curb_info.get("is_curb", False))
    curb_from_kink = bool(abs(dz) >= config.curb_height_thresh and slope >= config.curb_slope_thresh)
    is_curb = curb_from_kink if config.use_doc_kink_metrics else curb_from_interface
    curb_eval_mode = "kink_only" if config.use_doc_kink_metrics else "interface_only"
    used_curb_slope = float(slope) if config.use_doc_kink_metrics else float(curb_info.get("edge_slope", np.nan))
    is_ditch = bool(ditch_info.get("ditch_found", False))
    ambiguous = int(bool(is_curb and is_ditch))
    curb_override_slope_thresh = float(config.curb_override_slope_mult * config.curb_slope_thresh)
    curb_priority_override = bool(config.ditch_priority_over_curb and ambiguous and np.isfinite(used_curb_slope) and used_curb_slope >= curb_override_slope_thresh)
    dbg.update({"curb_from_interface": curb_from_interface, "curb_from_kink": curb_from_kink, "curb_eval_mode": curb_eval_mode, "is_curb": is_curb, "is_ditch": is_ditch, "priority_mode": "ditch_over_curb" if config.ditch_priority_over_curb else "curb_over_ditch", "priority_mode_effective": "curb_override_over_ditch" if curb_priority_override else ("ditch_over_curb" if config.ditch_priority_over_curb else "curb_over_ditch"), "curb_priority_override": curb_priority_override, "curb_priority_slope": used_curb_slope, "curb_override_slope_thresh": curb_override_slope_thresh, "ambiguous": ambiguous})

    curb_result = {"class": "CURB_NO_RAMP", "gap_m": gap, "dz_edge": curb_info.get("dz_edge", np.nan), "edge_slope": curb_info.get("edge_slope", np.nan), "dz_m": dz, "slope": slope, "dz_source": dz_source, "curb_source": curb_eval_mode, "curb_priority_override": curb_priority_override, "ambiguous": ambiguous}

    # Only build the ditch class payload after classify_ditch has actually
    # reported a ditch. Sparse/invalid ditch windows intentionally return early
    # without valley metrics such as valley_depth and vmin.
    def ditch_result() -> dict:
        return {"class": "DEPRESSED_DITCH", "depth_m": ditch_info["valley_depth"], "ditch_center_v": ditch_info["vmin"], "ditch_width50_m": ditch_info["valley_width_50"], "ditch_width30_m": ditch_info["valley_width_30"], "ditch_curv": ditch_info["valley_curv"], "gap_m": gap, "dz_edge": curb_info.get("dz_edge", np.nan), "edge_slope": curb_info.get("edge_slope", np.nan), "curb_candidate": bool(is_curb), "curb_priority_override": curb_priority_override, "dz_m": dz, "slope": slope, "dz_source": dz_source, "ambiguous": ambiguous}

    # Current notebook default is curb-over-ditch. The alternate mode is kept
    # because it is a real classifier option, but it is not enabled by default.
    if config.ditch_priority_over_curb:
        if curb_priority_override:
            return curb_result, dbg
        if is_ditch:
            return ditch_result(), dbg
        if is_curb:
            return curb_result, dbg
    else:
        if is_curb:
            return curb_result, dbg
        if is_ditch:
            return ditch_result(), dbg

    dbg["ramp_dz_source"] = dz_source
    return {"class": "RAMP", "dz_m": dz, "slope": slope, "transition_width_m": width, "kink1_v": kink1, "kink2_v": kink2, "gap_m": gap, "dz_source": dz_source, "curb_priority_override": curb_priority_override, "ambiguous": ambiguous}, dbg


def build_corridor_results(data: LoadedData, refined: RefinedTrajectory, config: PipelineConfig) -> list[dict]:
    """Extract raw lane/sidewalk cross-section arrays for every station.

    This is the script equivalent of the notebook's Cell 3. It does not
    classify anything; it only samples the corridor and stores the oriented
    lateral/elevation arrays plus masks that identify physical RIGHT and LEFT.
    The classifier consumes these station records in the next stage.
    """

    stations = np.arange(0.0, data.s[-1], config.step_size)
    results = []
    skipped = 0
    use_corr = bool(refined.enabled and refined.cs_x_corr is not None and refined.cs_y_corr is not None and refined.s_offsets is not None and len(refined.s_offsets) >= 2)
    s_corr_min = float(np.min(refined.s_offsets)) if use_corr else np.nan
    s_corr_max = float(np.max(refined.s_offsets)) if use_corr else np.nan
    print(f"Corridor loop: {len(stations):,} stations (step={config.step_size} m)")
    for s_target in stations:
        idx = int(np.argmin(np.abs(data.s - s_target)))

        # Prefer corrected geometry where refinement is valid. Outside the
        # correction domain, fall back to original geometry to avoid spline
        # extrapolation artifacts near segment ends.
        if use_corr and s_corr_min <= float(s_target) <= s_corr_max:
            p = np.array([float(refined.cs_x_corr(s_target)), float(refined.cs_y_corr(s_target))])
            heading_cs_x, heading_cs_y = refined.cs_x_corr, refined.cs_y_corr
            geom_source = "corr_spline"
        elif config.use_spline:
            p = np.array([float(data.cs_x(s_target)), float(data.cs_y(s_target))])
            heading_cs_x, heading_cs_y = data.cs_x, data.cs_y
            geom_source = "orig_spline"
        else:
            p = data.traj_xy[idx]
            heading_cs_x = heading_cs_y = None
            geom_source = "raw_traj"
        t_hat = compute_heading(data.traj_xy, data.s, idx, config.heading_window, float(s_target), heading_cs_x, heading_cs_y)
        if t_hat is None:
            skipped += 1
            continue
        n_hat = np.array([t_hat[1], -t_hat[0]])

        # Extract lane and sidewalk using the same station anchor and heading
        # so both surfaces share a common local v/z profile.
        v_lane_raw, z_lane = extract_cross_section(data.lane_xy, data.lane_z, data.lane_tree, p, t_hat, config.half_thick_u, config.v_min, config.v_max, config.query_radius)
        v_side_raw, z_side = extract_cross_section(data.side_xy, data.side_z, data.side_tree, p, t_hat, config.half_thick_u, config.v_min, config.v_max, config.query_radius)

        # Orientation stabilizes classifier assumptions. Physical masks preserve
        # actual RIGHT/LEFT identity for exports.
        v_lane, v_side, near_sign = orient_cross_section(v_lane_raw, v_side_raw)
        ml_r, ms_r, ml_l, ms_l = get_physical_side_masks(v_lane, v_side, near_sign)
        results.append({"s": float(s_target), "traj_source": geom_source, "p": p.copy(), "t_hat": t_hat.copy(), "n_hat": n_hat.copy(), "near_sign": near_sign, "v_lane": v_lane, "z_lane": z_lane, "v_side": v_side, "z_side": z_side, "ml_R": ml_r, "ms_R": ms_r, "ml_L": ml_l, "ms_L": ms_l})
    print(f"Done. {len(results):,} stations stored, {skipped:,} skipped (no heading).")
    return results


def classify_corridor_results(results: list[dict], config: PipelineConfig) -> list[dict]:
    """Run the rule-based classifier on both sides of every station.

    Each station already has masks for physical RIGHT and LEFT. This function
    slices lane/sidewalk arrays by those masks, classifies each side, attaches
    optional sidewalk-cluster metadata, and returns the enriched station list
    that drives every export.
    """

    classified = []
    for r in results:
        vl = r["v_lane"]
        zl = r["z_lane"]
        vs = r["v_side"]
        zs = r["z_side"]
        vl_r, zl_r = vl[r["ml_R"]], zl[r["ml_R"]]
        vs_r, zs_r = vs[r["ms_R"]], zs[r["ms_R"]]
        vl_l, zl_l = vl[r["ml_L"]], zl[r["ml_L"]]
        vs_l, zs_l = vs[r["ms_L"]], zs[r["ms_L"]]

        # Classify physical sides independently. A station can be ramp-like on
        # one side and separated/no-sidewalk on the other.
        res_r, dbg_r = classify_side(vl_r, zl_r, vs_r, zs_r, config)
        res_l, dbg_l = classify_side(vl_l, zl_l, vs_l, zs_l, config)

        # Cluster metadata is attached after side classification because the
        # cluster labels inherit the side-level result unless split is enabled.
        clusters_r = build_sidewalk_clusters(vs_r, zs_r, len(vl_r) == 0 or np.median(vl_r) >= 0, res_r["class"], config)
        clusters_l = build_sidewalk_clusters(vs_l, zs_l, len(vl_l) == 0 or np.median(vl_l) >= 0, res_l["class"], config)
        res_r = {**res_r, "cluster_count": len(clusters_r), "meaningful_cluster_count": int(sum(c.get("is_meaningful", False) for c in clusters_r)), "detached_cluster_count": int(sum(c.get("is_detached", False) for c in clusters_r))}
        res_l = {**res_l, "cluster_count": len(clusters_l), "meaningful_cluster_count": int(sum(c.get("is_meaningful", False) for c in clusters_l)), "detached_cluster_count": int(sum(c.get("is_detached", False) for c in clusters_l))}
        if dbg_r is not None:
            dbg_r["sidewalk_clusters"] = clusters_r
        if dbg_l is not None:
            dbg_l["sidewalk_clusters"] = clusters_l
        classified.append({**r, "class_R": res_r["class"], "class_L": res_l["class"], "meta_R": res_r, "meta_L": res_l, "dbg_R": dbg_r, "dbg_L": dbg_l, "side_clusters_R": clusters_r, "side_clusters_L": clusters_l})

    cnt_r = Counter(r["class_R"] for r in classified)
    cnt_l = Counter(r["class_L"] for r in classified)
    print(f"Classified {len(classified):,} stations")
    print("RIGHT side:", dict(sorted(cnt_r.items())))
    print("LEFT side :", dict(sorted(cnt_l.items())))
    return classified


def output_base_path(config: PipelineConfig) -> Path:
    """Return the output prefix path used for all Stage 1 products."""

    out_dir = config.output_dir if config.output_dir is not None else config.traj_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / config.traj_csv.stem


def export_point_products(data: LoadedData, classified: list[dict], config: PipelineConfig) -> tuple[PipelineOutputs, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export trajectory, sidewalk, and lane CSV/ASC products.

    Three point products are written:

    - trajectory rows: one row per station, colored by dominant side class;
    - sidewalk rows: one row per exported sidewalk point, labeled by side class
      or optional cluster label;
    - lane rows: one row per exported lane point, labeled with the station-side
      class so Stage 2 can rebuild lane/sidewalk profiles from matching labels.

    The notebook stores local lateral coordinates during extraction. Exporting
    point products requires projecting those local ``v`` values back to world XY
    with ``p + v_raw * n_hat``. ``near_sign`` is used to undo any orientation
    flip that was applied for classification.
    """

    base = output_base_path(config)
    cand_z = ["z", "Z", "alt", "altitude", "elevation", "Elevation", "height", "Height", "utm_z"]
    zcol = next((c for c in cand_z if c in data.traj_df.columns), None)
    traj_z = data.traj_df[zcol].to_numpy(float) if zcol else None

    # A) Trajectory export: one station anchor point with dominant RGB. This is
    # useful for quickly viewing the corridor-level classification in CC.
    traj_rows = []
    for station in classified:
        traj_idx = int(np.argmin(np.abs(data.s - station["s"])))
        tx, ty = map(float, station["p"])
        tz = float(traj_z[traj_idx]) if traj_z is not None else 0.0
        class_r = station["class_R"]
        class_l = station["class_L"]
        dom = dominant_class(class_r, class_l)
        cr, cg, cb = CLASS_COLOR.get(dom, CLASS_COLOR["UNKNOWN"])
        dom_meta = station["meta_R"] if PRIORITY.get(class_r, 99) <= PRIORITY.get(class_l, 99) else station["meta_L"]
        traj_rows.append({"//X": tx, "Y": ty, "Z": tz, "R": cr, "G": cg, "B": cb, "s_m": round(station["s"], 3), "class_RIGHT": CLASS_CODE.get(class_r, 5), "class_LEFT": CLASS_CODE.get(class_l, 5), "label_R": class_r, "label_L": class_l, "ambiguous_RIGHT": int(station["meta_R"].get("ambiguous", 0)), "ambiguous_LEFT": int(station["meta_L"].get("ambiguous", 0)), "dominant": dom, "hex_R": rgb_to_hex(*CLASS_COLOR.get(class_r, CLASS_COLOR["UNKNOWN"])), "hex_L": rgb_to_hex(*CLASS_COLOR.get(class_l, CLASS_COLOR["UNKNOWN"])), "gap_m": round(float(dom_meta.get("gap_m", np.nan)), 4), "dz_m": round(float(dom_meta.get("dz_m", np.nan)), 4), "slope": round(float(dom_meta.get("slope", np.nan)), 4), "depth_m": round(float(dom_meta.get("depth_m", np.nan)), 4)})
    df_traj = pd.DataFrame(traj_rows)

    cluster_role_code = {"primary": 0, "detached": 1, "noise": 2}
    sw_rows = []
    lane_rows = []

    # B/C) Point exports: rebuild world coordinates from each station-local
    # lateral coordinate. This keeps the original point cloud geometry while
    # adding station-side class labels and colors.
    for station in classified:
        p = station["p"]
        n_hat = station["n_hat"]
        near_sign = station["near_sign"]
        vs = station["v_side"]
        zs = station["z_side"]
        assignments = [("RIGHT", "R", station["class_R"], station["ms_R"], station.get("side_clusters_R", [])), ("LEFT", "L", station["class_L"], station["ms_L"], station.get("side_clusters_L", []))]
        for side_label, side_key, side_class, mask, clusters in assignments:
            if mask.sum() == 0:
                continue
            v_side_side = vs[mask]
            z_side_side = zs[mask]
            ambiguous = int(station[f"meta_{side_key}"].get("ambiguous", 0))
            v_lane_side = station["v_lane"][station[f"ml_{side_key}"]]
            z_lane_side = station["z_lane"][station[f"ml_{side_key}"]]
            if len(v_lane_side) > 0:
                # Undo any classification-time v flip before projecting back to
                # world XY. Without this, points would mirror across the anchor.
                v_lane_raw = v_lane_side * near_sign
                lane_world_x = p[0] + v_lane_raw * n_hat[0]
                lane_world_y = p[1] + v_lane_raw * n_hat[1]
                lane_cr, lane_cg, lane_cb = CLASS_COLOR.get(side_class, CLASS_COLOR["UNKNOWN"])
                lane_hex = rgb_to_hex(lane_cr, lane_cg, lane_cb)
                for wx, wy, wz, vl_val in zip(lane_world_x, lane_world_y, z_lane_side, v_lane_side):
                    lane_rows.append({"x": wx, "y": wy, "z": wz, "R": lane_cr, "G": lane_cg, "B": lane_cb, "s_m": round(float(station["s"]), 3), "v_m": round(float(vl_val), 4), "side": side_label, "class": CLASS_CODE.get(side_class, 5), "label": side_class, "color": lane_hex, "ambiguous": ambiguous})
            if not clusters:
                # Keep export shape stable even when there are no split clusters
                # for a side: every point still belongs to a primary cluster.
                clusters = [{"cluster_id": 0, "cluster_role": "primary", "cluster_label": side_class, "point_idx_local": np.arange(len(v_side_side), dtype=int), "v_min": float(np.min(v_side_side)) if len(v_side_side) else np.nan, "v_max": float(np.max(v_side_side)) if len(v_side_side) else np.nan, "gap_from_previous_cluster_m": np.nan}]
            for cluster in clusters:
                idx_local = np.asarray(cluster.get("point_idx_local", []), dtype=int)
                if len(idx_local) == 0:
                    continue
                v_sw = v_side_side[idx_local]
                z_sw = z_side_side[idx_local]
                # Sidewalk world coordinates use the same inverse projection as
                # lane points, but labels may come from cluster metadata.
                v_raw = v_sw * near_sign
                world_x = p[0] + v_raw * n_hat[0]
                world_y = p[1] + v_raw * n_hat[1]
                class_name = cluster.get("cluster_label", side_class)
                cluster_role = cluster.get("cluster_role", "primary")
                cr, cg, cb = CLASS_COLOR.get(class_name, CLASS_COLOR["UNKNOWN"])
                hex_col = rgb_to_hex(cr, cg, cb)
                cluster_v_min = safe_float(cluster.get("v_min", np.nan))
                cluster_v_max = safe_float(cluster.get("v_max", np.nan))
                cluster_gap_prev = safe_float(cluster.get("gap_from_previous_cluster_m", np.nan))
                for wx, wy, wz, vs_val in zip(world_x, world_y, z_sw, v_sw):
                    sw_rows.append({"x": wx, "y": wy, "z": wz, "R": cr, "G": cg, "B": cb, "s_m": round(float(station["s"]), 3), "v_m": round(float(vs_val), 4), "side": side_label, "class": CLASS_CODE.get(class_name, 5), "label": class_name, "color": hex_col, "ambiguous": ambiguous, "cluster_id": int(cluster.get("cluster_id", -1)), "cluster_role": cluster_role, "cluster_role_code": cluster_role_code.get(cluster_role, 99), "cluster_label": class_name, "cluster_v_min_m": round(cluster_v_min, 4) if np.isfinite(cluster_v_min) else np.nan, "cluster_v_max_m": round(cluster_v_max, 4) if np.isfinite(cluster_v_max) else np.nan, "cluster_gap_prev_m": round(cluster_gap_prev, 4) if np.isfinite(cluster_gap_prev) else np.nan})

    df_sw = pd.DataFrame(sw_rows)
    df_lane = pd.DataFrame(lane_rows)

    # Output names intentionally match the notebook naming convention so the
    # downstream dataset and prediction notebooks can consume them unchanged.
    outputs = PipelineOutputs(
        traj_csv=base.with_name(base.name + "_traj.csv"),
        sidewalk_csv=base.with_name(base.name + "_sidewalk.csv"),
        lane_csv=base.with_name(base.name + "_lane.csv"),
        traj_asc=base.with_name(base.name + "_traj_CC.asc"),
        sidewalk_asc=base.with_name(base.name + "_sidewalk_CC.asc"),
        lane_asc=base.with_name(base.name + "_lane_CC.asc"),
        metrics_csv=base.with_name(base.name + "_station_side_metrics.csv"),
        metrics_json=base.with_name(base.name + "_station_side_metrics.json"),
        run_params_txt=base.with_name(base.name + "_run_params.txt"),
    )

    # CloudCompare ASC files use space-separated columns with //X as the first
    # column name. RGB columns make the class colors visible immediately.
    with outputs.traj_asc.open("w", encoding="utf-8") as f:
        f.write("//X Y Z R G B s_m class_RIGHT class_LEFT ambiguous_RIGHT ambiguous_LEFT\n")
        df_traj[["//X", "Y", "Z", "R", "G", "B", "s_m", "class_RIGHT", "class_LEFT", "ambiguous_RIGHT", "ambiguous_LEFT"]].to_csv(f, sep=" ", index=False, header=False)
    df_traj.to_csv(outputs.traj_csv, index=False)

    df_cc_sw = df_sw[["x", "y", "z", "R", "G", "B", "s_m", "v_m", "side", "class", "cluster_id", "cluster_role_code", "ambiguous"]].copy()
    df_cc_sw["side"] = df_cc_sw["side"].map(SIDE_TO_INT)
    df_cc_sw = df_cc_sw.rename(columns={"x": "//X"})
    with outputs.sidewalk_asc.open("w", encoding="utf-8") as f:
        f.write("//X Y Z R G B s_m v_m side class cluster_id cluster_role ambiguous\n")
        df_cc_sw.to_csv(f, sep=" ", index=False, header=False)
    df_sw.to_csv(outputs.sidewalk_csv, index=False)

    df_cc_lane = df_lane[["x", "y", "z", "R", "G", "B", "s_m", "v_m", "side", "class", "ambiguous"]].copy()
    df_cc_lane["side"] = df_cc_lane["side"].map(SIDE_TO_INT)
    df_cc_lane = df_cc_lane.rename(columns={"x": "//X"})
    with outputs.lane_asc.open("w", encoding="utf-8") as f:
        f.write("//X Y Z R G B s_m v_m side class ambiguous\n")
        df_cc_lane.to_csv(f, sep=" ", index=False, header=False)
    df_lane.to_csv(outputs.lane_csv, index=False)
    write_run_params(outputs.run_params_txt, config)
    print(f"Exported point products: traj={len(df_traj):,}, sidewalk={len(df_sw):,}, lane={len(df_lane):,}")
    return outputs, df_traj, df_sw, df_lane


def json_safe(value):
    """Convert NumPy scalars/NaNs into JSON-safe values."""

    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def export_station_side_metrics(classified: list[dict], outputs: PipelineOutputs, config: PipelineConfig) -> pd.DataFrame:
    """Export one structured metrics row per station-side.

    This is the script equivalent of notebook Cell 5.5. The output is keyed by
    rounded ``s_m`` plus ``side`` so Stage 2 can merge scalar engineered
    features into one CNN dataset row per station-side. Metrics are exported
    even when they were not the winning class so later experiments can use the
    same CSV without re-running the rule-based classifier.
    """

    rows = []
    for r in classified:
        for side_key, side_label in [("R", "RIGHT"), ("L", "LEFT")]:
            meta = r.get(f"meta_{side_key}", {}) or {}
            dbg = r.get(f"dbg_{side_key}", {}) or {}
            label = meta.get("class", r.get(f"class_{side_key}", "UNKNOWN"))

            # meta contains compact final-class fields; dbg contains richer
            # profile/interface metrics. Prefer meta for final labels and use
            # dbg as fallback for class-independent engineered metrics.
            rows.append({
                "source_file": str(config.traj_csv),
                "s_m": json_safe(round(float(r["s"]), 3)),
                "s_key": json_safe(round(float(r["s"]), 3)),
                "side": side_label,
                "label": label,
                "class": json_safe(CLASS_CODE.get(label, 5)),
                "kink_dz": json_safe(meta.get("dz_m", np.nan)),
                "kink_slope": json_safe(meta.get("slope", np.nan)),
                "kink_width_m": json_safe(meta.get("transition_width_m", np.nan)),
                "kink1_v": json_safe(meta.get("kink1_v", dbg.get("v0", np.nan))),
                "kink2_v": json_safe(meta.get("kink2_v", dbg.get("v1", np.nan))),
                "dz_source": json_safe(meta.get("dz_source", dbg.get("ramp_dz_source", np.nan))),
                "gap_m": json_safe(meta.get("gap_m", np.nan)),
                "lane_edge_v": json_safe(dbg.get("lane_edge_v", np.nan)),
                "side_edge_v": json_safe(dbg.get("side_edge_v", np.nan)),
                "interface_center_v": json_safe(dbg.get("interface_center_v", np.nan)),
                "dz_edge": json_safe(meta.get("dz_edge", dbg.get("dz_edge", np.nan))),
                "edge_slope": json_safe(meta.get("edge_slope", dbg.get("edge_slope", np.nan))),
                "curb_from_interface": json_safe(dbg.get("curb_from_interface", np.nan)),
                "curb_from_kink": json_safe(dbg.get("curb_from_kink", np.nan)),
                "curb_eval_mode": json_safe(dbg.get("curb_eval_mode", np.nan)),
                "curb_priority_override": json_safe(meta.get("curb_priority_override", dbg.get("curb_priority_override", np.nan))),
                "ditch_depth_m": json_safe(meta.get("depth_m", np.nan)),
                "ditch_center_v": json_safe(meta.get("ditch_center_v", dbg.get("vmin", np.nan))),
                "ditch_width50_m": json_safe(meta.get("ditch_width50_m", np.nan)),
                "ditch_width30_m": json_safe(meta.get("ditch_width30_m", np.nan)),
                "ditch_curv": json_safe(meta.get("ditch_curv", np.nan)),
                "ambiguous": json_safe(meta.get("ambiguous", dbg.get("ambiguous", np.nan))),
                "cluster_count": json_safe(meta.get("cluster_count", np.nan)),
                "meaningful_cluster_count": json_safe(meta.get("meaningful_cluster_count", np.nan)),
                "detached_cluster_count": json_safe(meta.get("detached_cluster_count", np.nan)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(outputs.metrics_csv, index=False)
    with outputs.metrics_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Exported station-side metrics: {len(df):,} rows")
    return df


def save_las(path: Path, x, y, z, r, g, b) -> None:
    """Save XYZ plus RGB as a LAS 1.4 point-format-2 file."""

    header = laspy.LasHeader(point_format=2, version="1.4")
    las = laspy.LasData(header=header)
    las.x = np.asarray(x, dtype=float)
    las.y = np.asarray(y, dtype=float)
    las.z = np.asarray(z, dtype=float)
    las.red = np.asarray(r, dtype=np.uint16) * 256
    las.green = np.asarray(g, dtype=np.uint16) * 256
    las.blue = np.asarray(b, dtype=np.uint16) * 256
    las.write(path)
    print(f"  LAS saved -> {path} ({len(las.x):,} pts)")


def export_las_outputs(outputs: PipelineOutputs, df_traj: pd.DataFrame, df_sw: pd.DataFrame, df_lane: pd.DataFrame) -> PipelineOutputs:
    """Write optional LAS versions of the trajectory/sidewalk/lane outputs."""

    outputs.traj_las = outputs.traj_asc.with_suffix(".las")
    outputs.sidewalk_las = outputs.sidewalk_asc.with_suffix(".las")
    outputs.lane_las = outputs.lane_asc.with_suffix(".las")
    save_las(outputs.traj_las, df_traj["//X"], df_traj["Y"], df_traj["Z"], df_traj["R"], df_traj["G"], df_traj["B"])
    save_las(outputs.sidewalk_las, df_sw["x"], df_sw["y"], df_sw["z"], df_sw["R"], df_sw["G"], df_sw["B"])
    save_las(outputs.lane_las, df_lane["x"], df_lane["y"], df_lane["z"], df_lane["R"], df_lane["G"], df_lane["B"])
    return outputs


def write_run_params(path: Path, config: PipelineConfig) -> None:
    """Write a simple parameter snapshot beside the output products."""

    lines = ["Ramp_Detection run parameter snapshot", "=" * 42]
    for key, value in vars(config).items():
        lines.append(f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    """Run the full core rule-based corridor pipeline and write outputs.

    The orchestration is deliberately linear and mirrors the notebook cell flow:

    1. ``load_inputs`` corresponds to Cell 1 setup without the sanity plot.
    2. ``refine_trajectory`` corresponds to Cell 2.5 without diagnostic plots.
    3. ``build_corridor_results`` corresponds to Cell 3.
    4. ``classify_corridor_results`` corresponds to Cell 4.
    5. export helpers correspond to Cells 5, 5.5, and optionally 6.

    Keeping the steps explicit makes it easier to later add batch processing
    around this function without hiding any state transitions.
    """

    data = load_inputs(config)
    refined = refine_trajectory(data, config)
    results = build_corridor_results(data, refined, config)
    classified = classify_corridor_results(results, config)
    outputs, df_traj, df_sw, df_lane = export_point_products(data, classified, config)
    export_station_side_metrics(classified, outputs, config)
    if config.write_las:
        outputs = export_las_outputs(outputs, df_traj, df_sw, df_lane)
    print("Pipeline complete.")
    return outputs


def parse_voxel(value: str) -> float | None:
    """Parse --voxel, allowing 'none' to disable voxel downsampling."""

    if value.lower() in {"none", "null", "off", "false"}:
        return None
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for one-segment processing."""

    parser = argparse.ArgumentParser(description="Run the core updated_ditch rule-based ramp detection pipeline.")
    parser.add_argument("--sidewalk-las", required=True, type=Path, help="Input sidewalk LAS/LAZ file.")
    parser.add_argument("--lane-las", required=True, type=Path, help="Input lane/road LAS/LAZ file.")
    parser.add_argument("--traj-csv", required=True, type=Path, help="Input trajectory CSV file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory. Defaults to the trajectory CSV folder.")
    parser.add_argument("--no-spline", action="store_true", help="Use raw trajectory samples instead of cubic splines.")
    parser.add_argument("--no-refinement", action="store_true", help="Disable lane-centre trajectory refinement.")
    parser.add_argument("--write-las", action="store_true", help="Also write LAS versions of the exported point products.")
    parser.add_argument("--voxel", type=parse_voxel, default=0.05, help="Voxel size in metres, or 'none' to disable. Default: 0.05")
    parser.add_argument("--step-size", type=float, default=0.2, help="Station spacing in metres. Default: 0.2")
    parser.add_argument("--half-thick-u", type=float, default=0.4, help="Half-thickness of cross-section slab in metres. Default: 0.4")
    parser.add_argument("--v-min", type=float, default=-15.0, help="Minimum lateral coordinate for slab extraction. Default: -15")
    parser.add_argument("--v-max", type=float, default=15.0, help="Maximum lateral coordinate for slab extraction. Default: 15")
    parser.add_argument("--query-radius", type=float, default=35.0, help="KD-tree radius query distance in metres. Default: 35")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Create a PipelineConfig from parsed command-line arguments."""

    return PipelineConfig(
        sidewalk_las=args.sidewalk_las,
        lane_las=args.lane_las,
        traj_csv=args.traj_csv,
        output_dir=args.output_dir,
        use_spline=not args.no_spline,
        use_traj_refinement=not args.no_refinement,
        write_las=args.write_las,
        voxel=args.voxel,
        step_size=args.step_size,
        half_thick_u=args.half_thick_u,
        v_min=args.v_min,
        v_max=args.v_max,
        query_radius=args.query_radius,
    )


def validate_inputs(config: PipelineConfig) -> None:
    """Fail early if required input paths do not exist."""

    for label, path in [("sidewalk LAS", config.sidewalk_las), ("lane LAS", config.lane_las), ("trajectory CSV", config.traj_csv)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    validate_inputs(config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
