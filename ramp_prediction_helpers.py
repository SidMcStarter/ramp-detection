from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


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

INV_MAP = {v: k for k, v in CLASS_CODE.items()}
SIDE_TO_INT = {"RIGHT": 0, "LEFT": 1}


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def normalize_side_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def add_station_side_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["side"] = normalize_side_column(out["side"])
    out["_s_key"] = pd.to_numeric(out["s_m"], errors="coerce").round(3)
    return out


def mode_or_first(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


# -----------------------------------------------------------------------------
# Stage 1 feature helpers copied/refactored from updated_ditch_latest_version.
# -----------------------------------------------------------------------------

BIN_SIZE = 0.08
MIN_PTS = 3
SG_WIN = 11
SG_POLY = 4
MIN_SW_PTS = 50
UNKNOWN_MIN_ALL_PTS = 100
GAP_MAX = 2.0
EDGE_BAND = 0.35
CURB_HEIGHT_THRESH = 0.05
CURB_SLOPE_THRESH = 0.1
CURB_OVERRIDE_SLOPE_MULT = 2.5
USE_DOC_KINK_METRICS = True
INTERFACE_KINK_BUFFER = 0.75
USE_DITCH_BUFFER = True
DITCH_BUFFER = 3.0
DITCH_PRIORITY_OVER_CURB = True


def build_binned_profile(
    v: np.ndarray,
    z: np.ndarray,
    bin_size: float = BIN_SIZE,
    min_pts: int = MIN_PTS,
) -> tuple[np.ndarray, np.ndarray]:
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


def smooth_savgol(z: np.ndarray, win: int = SG_WIN, poly: int = SG_POLY) -> np.ndarray:
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
    v = np.asarray(v)
    z = np.asarray(z)
    ok = np.isfinite(v) & np.isfinite(z)
    if np.sum(ok) < 2:
        return z.copy(), (0.0, 0.0)
    a, b = np.polyfit(v[ok], z[ok], 1)
    return z - (a * v + b), (float(a), float(b))


def build_full_profile(
    v_all: np.ndarray,
    z_all: np.ndarray,
    bin_size: float = BIN_SIZE,
    min_pts: int = MIN_PTS,
    sg_win: int = SG_WIN,
    sg_poly: int = SG_POLY,
) -> dict | None:
    v_all = np.asarray(v_all)
    z_all = np.asarray(z_all)
    ok = np.isfinite(v_all) & np.isfinite(z_all)
    v_all = v_all[ok]
    z_all = z_all[ok]

    if len(v_all) < 20:
        return None

    order = np.argsort(v_all)
    v_all = v_all[order]
    z_all = z_all[order]

    v_ctr, z_med = build_binned_profile(v_all, z_all, bin_size, min_pts)
    if len(v_ctr) == 0:
        return None

    ok2 = np.isfinite(z_med)
    if np.sum(ok2) < 12:
        return None

    z_fill = z_med.copy()
    z_fill[~ok2] = np.interp(v_ctr[~ok2], v_ctr[ok2], z_fill[ok2])
    z_s = smooth_savgol(z_fill, sg_win, sg_poly)
    z_dt, trend = detrend_profile(v_ctr, z_s)
    d1 = np.gradient(z_dt, bin_size)
    d2 = np.gradient(d1, bin_size)

    return {
        "v_ctr": v_ctr,
        "z_med": z_med,
        "z_fill": z_fill,
        "z_s": z_s,
        "z_dt": z_dt,
        "d1": d1,
        "d2": d2,
        "trend_coef": trend,
    }


def pick_two_kinks(
    v_ctr: np.ndarray,
    z_s: np.ndarray,
    bin_size: float,
    v_lo: float | None = None,
    v_hi: float | None = None,
    min_bins: int = 5,
) -> tuple[float, float, int, int, np.ndarray, str, str]:
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


def estimate_side_interface(
    v_lane_side: np.ndarray,
    z_lane_side: np.ndarray,
    v_side_side: np.ndarray,
    z_side_side: np.ndarray,
    edge_band: float = EDGE_BAND,
) -> dict | None:
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


def classify_curb_from_interface(
    interface: dict | None,
    curb_height_thresh: float = CURB_HEIGHT_THRESH,
    curb_slope_thresh: float = CURB_SLOPE_THRESH,
) -> dict:
    if interface is None:
        return {"is_curb": False, "curb_reason": "no_interface"}

    dz = abs(interface["dz_edge"])
    slope = interface["edge_slope"]
    is_curb = dz >= curb_height_thresh and slope >= curb_slope_thresh
    return {
        "is_curb": bool(is_curb),
        "curb_reason": "edge_step_detected" if is_curb else "no_clear_edge_step",
        **interface,
    }


def summarize_profile_geometry(v_ctr: np.ndarray, z_dt: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> dict:
    i_min = int(np.argmin(z_dt))
    i_max = int(np.argmax(z_dt))
    zmin = float(z_dt[i_min])
    zmax = float(z_dt[i_max])
    n_end = max(3, len(z_dt) // 5)
    z_l = float(np.median(z_dt[:n_end]))
    z_r = float(np.median(z_dt[-n_end:]))
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
        thresh50 = zmin + 0.5 * valley_depth
        idx50 = np.where(z_dt <= thresh50)[0]
        valley_width_50 = float(v_ctr[idx50[-1]] - v_ctr[idx50[0]]) if len(idx50) >= 2 else np.nan
        thresh30 = zmin + 0.3 * valley_depth
        idx30 = np.where(z_dt <= thresh30)[0]
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


def classify_ditch(
    prof: dict | None,
    use_buffer: bool = USE_DITCH_BUFFER,
    v_center: float = np.nan,
    ditch_buffer: float = DITCH_BUFFER,
    min_bins: int = 5,
) -> dict:
    out = {
        "ditch_found": False,
        "geom_class": "UNKNOWN",
        "note": "sparse_profile",
        "ditch_eval_mode": "interface_buffer" if use_buffer else "full_profile",
        "ditch_window_v0": np.nan,
        "ditch_window_v1": np.nan,
        "ditch_window_center_v": float(v_center) if np.isfinite(v_center) else np.nan,
        "ditch_window_note": "buffer_requested" if use_buffer else "buffer_disabled",
        "ditch_window_bins": np.nan,
    }
    if prof is None:
        return out

    v_use = prof["v_ctr"]
    z_use = prof["z_dt"]
    d1_use = prof["d1"]
    d2_use = prof["d2"]

    if use_buffer:
        if not np.isfinite(v_center):
            out.update({"geom_class": "NOT_DITCH", "note": "ditch_buffer_center_unavailable", "ditch_window_note": "ditch_buffer_center_unavailable"})
            return out
        v_lo = float(v_center - ditch_buffer)
        v_hi = float(v_center + ditch_buffer)
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


def classify_side(
    v_lane: np.ndarray,
    z_lane: np.ndarray,
    v_side: np.ndarray,
    z_side: np.ndarray,
    bin_size: float = BIN_SIZE,
    min_pts: int = MIN_PTS,
    sg_win: int = SG_WIN,
    sg_poly: int = SG_POLY,
    min_sw_pts: int = MIN_SW_PTS,
    gap_max: float = GAP_MAX,
    edge_band: float = EDGE_BAND,
    curb_height_thresh: float = CURB_HEIGHT_THRESH,
    curb_slope_thresh: float = CURB_SLOPE_THRESH,
) -> tuple[dict, dict | None]:
    def _empty(cls, gap=np.nan):
        return {
            "class": cls,
            "gap_m": gap,
            "dz_m": np.nan,
            "slope": np.nan,
            "transition_width_m": np.nan,
            "kink1_v": np.nan,
            "kink2_v": np.nan,
            "ambiguous": 0,
        }, None

    v_lane = np.asarray(v_lane, dtype=float)
    z_lane = np.asarray(z_lane, dtype=float)
    v_side = np.asarray(v_side, dtype=float)
    z_side = np.asarray(z_side, dtype=float)
    on_right = len(v_lane) == 0 or np.median(v_lane) >= 0

    if len(v_side) < min_sw_pts:
        return _empty("NO_SIDEWALK")

    gap = np.nan
    if len(v_lane) > 30 and len(v_side) > 30:
        if on_right:
            lane_edge = np.percentile(v_lane, 95)
            sw_edge = np.percentile(v_side, 5)
        else:
            lane_edge = np.percentile(v_lane, 5)
            sw_edge = np.percentile(v_side, 95)
        gap = float(abs(sw_edge - lane_edge))
        if gap > gap_max:
            return _empty("SEPARATED_NO_RAMP", gap)

    v_all = np.concatenate([v_lane, v_side])
    z_all = np.concatenate([z_lane, z_side])
    if len(v_all) < UNKNOWN_MIN_ALL_PTS:
        return _empty("UNKNOWN", gap)

    prof = build_full_profile(v_all, z_all, bin_size, min_pts, sg_win, sg_poly)
    if prof is None:
        return _empty("UNKNOWN", gap)

    v_ctr = prof["v_ctr"]
    z_s = prof["z_s"]
    interface = estimate_side_interface(v_lane, z_lane, v_side, z_side, edge_band)
    curb_info = classify_curb_from_interface(interface, curb_height_thresh, curb_slope_thresh)

    interface_center_v = np.nan
    v_lo = v_hi = None
    if interface is not None and np.isfinite(interface.get("lane_edge_v", np.nan)) and np.isfinite(interface.get("side_edge_v", np.nan)):
        interface_center_v = 0.5 * float(interface["lane_edge_v"] + interface["side_edge_v"])
        v_lo = interface_center_v - INTERFACE_KINK_BUFFER
        v_hi = interface_center_v + INTERFACE_KINK_BUFFER

    kink1, kink2, i_k1, i_k2, d2_kink, kink_mode, kink_note = pick_two_kinks(v_ctr, z_s, bin_size, v_lo=v_lo, v_hi=v_hi)
    v0, v1 = sorted([kink1, kink2])
    width = float(v1 - v0)
    i0 = int(np.argmin(np.abs(v_ctr - v0)))
    i1 = int(np.argmin(np.abs(v_ctr - v1)))
    dz = float(z_s[i1] - z_s[i0]) if USE_DOC_KINK_METRICS else float(prof["z_dt"][i1] - prof["z_dt"][i0])
    dz_source = "smoothed_kink_bounds" if USE_DOC_KINK_METRICS else "detrended_kink_bounds"
    slope = abs(dz) / max(width, 1e-9)

    dbg = {
        **prof,
        "d2": d2_kink,
        "v0": v0,
        "v1": v1,
        "kink1": kink1,
        "kink2": kink2,
        "i_k1": i_k1,
        "i_k2": i_k2,
        "kink_mode": kink_mode,
        "kink_note": kink_note,
        "kink_window_v0": v_lo,
        "kink_window_v1": v_hi,
        "interface_center_v": interface_center_v,
        "dz_kink": dz,
        "slope_kink": slope,
        "dz_source": dz_source,
    }
    dbg.update(curb_info)
    if interface is not None:
        dbg.update(interface)

    if width <= 0.05:
        return {
            "class": "UNKNOWN",
            "note": "no_clear_kinks",
            "gap_m": gap,
            "dz_m": np.nan,
            "slope": np.nan,
            "transition_width_m": width,
            "kink1_v": kink1,
            "kink2_v": kink2,
            "kink_mode": kink_mode,
            "kink_note": kink_note,
            "ambiguous": 0,
        }, dbg

    ditch_info = classify_ditch(prof, use_buffer=USE_DITCH_BUFFER, v_center=interface_center_v, ditch_buffer=DITCH_BUFFER)
    dbg.update(ditch_info)

    curb_from_interface = bool(curb_info.get("is_curb", False))
    curb_from_kink = bool(abs(dz) >= curb_height_thresh and slope >= curb_slope_thresh)
    is_curb = curb_from_kink if USE_DOC_KINK_METRICS else curb_from_interface
    curb_eval_mode = "kink_only" if USE_DOC_KINK_METRICS else "interface_only"
    used_curb_slope = float(slope) if USE_DOC_KINK_METRICS else float(curb_info.get("edge_slope", np.nan))
    is_ditch = bool(ditch_info.get("ditch_found", False))
    ambiguous = int(bool(is_curb and is_ditch))
    curb_priority_override = bool(
        DITCH_PRIORITY_OVER_CURB
        and ambiguous
        and np.isfinite(used_curb_slope)
        and used_curb_slope >= CURB_OVERRIDE_SLOPE_MULT * curb_slope_thresh
    )

    dbg.update({
        "curb_from_interface": curb_from_interface,
        "curb_from_kink": curb_from_kink,
        "curb_eval_mode": curb_eval_mode,
        "is_curb": is_curb,
        "is_ditch": is_ditch,
        "curb_priority_override": curb_priority_override,
        "ambiguous": ambiguous,
    })

    common = {
        "gap_m": gap,
        "dz_edge": curb_info.get("dz_edge", np.nan),
        "edge_slope": curb_info.get("edge_slope", np.nan),
        "dz_m": dz,
        "slope": slope,
        "dz_source": dz_source,
        "curb_priority_override": curb_priority_override,
        "ambiguous": ambiguous,
    }
    if DITCH_PRIORITY_OVER_CURB and curb_priority_override:
        return {"class": "CURB_NO_RAMP", "curb_source": curb_eval_mode, **common}, dbg
    if DITCH_PRIORITY_OVER_CURB and is_ditch:
        return {
            "class": "DEPRESSED_DITCH",
            "depth_m": ditch_info["valley_depth"],
            "ditch_center_v": ditch_info["vmin"],
            "ditch_width50_m": ditch_info["valley_width_50"],
            "ditch_width30_m": ditch_info["valley_width_30"],
            "ditch_curv": ditch_info["valley_curv"],
            "curb_candidate": bool(is_curb),
            **common,
        }, dbg
    if is_curb:
        return {"class": "CURB_NO_RAMP", "curb_source": curb_eval_mode, **common}, dbg
    if (not DITCH_PRIORITY_OVER_CURB) and is_ditch:
        return {
            "class": "DEPRESSED_DITCH",
            "depth_m": ditch_info["valley_depth"],
            "ditch_center_v": ditch_info["vmin"],
            "ditch_width50_m": ditch_info["valley_width_50"],
            "ditch_width30_m": ditch_info["valley_width_30"],
            "ditch_curv": ditch_info["valley_curv"],
            "curb_candidate": bool(is_curb),
            **common,
        }, dbg

    return {
        "class": "RAMP",
        "transition_width_m": width,
        "kink1_v": kink1,
        "kink2_v": kink2,
        **common,
    }, dbg


# -----------------------------------------------------------------------------
# Stage 2 model-input profile helpers.
# -----------------------------------------------------------------------------

FULL_PROFILE_N_BINS = 192
SIDEWALK_EDGE_N_BINS = 25
USE_SIDEWALK_EDGE_BUFFER = False
SIDEWALK_EDGE_BUFFER_M = 1.0
N_BINS = SIDEWALK_EDGE_N_BINS if USE_SIDEWALK_EDGE_BUFFER else FULL_PROFILE_N_BINS
MIN_PTS_PER_BIN = 3
PROFILE_BIN_SIZE = 0.08
PROFILE_SG_WINDOW = 11
PROFILE_SG_POLY = 4
PROFILE_COLS = [f"z_{i:03d}" for i in range(N_BINS)]
GAP_COLS = [f"gap_{i:03d}" for i in range(N_BINS)]
PAD_COLS = [f"pad_{i:03d}" for i in range(N_BINS)]
FEATURE_COLS = ["feat_kink_dz", "feat_kink_slope"]


def prepare_profile_arrays(v, z) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)
    keep = np.isfinite(v) & np.isfinite(z)
    return np.abs(v[keep]), z[keep]


def bin_profile(
    v,
    z,
    bin_size: float = PROFILE_BIN_SIZE,
    n_bins: int = N_BINS,
    min_pts_per_bin: int = MIN_PTS_PER_BIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, z = prepare_profile_arrays(v, z)
    keep = (v >= 0.0) & (v < n_bins * bin_size)
    v = v[keep]
    z = z[keep]
    z_bins = np.full(n_bins, np.nan, dtype=float)
    mask = np.zeros(n_bins, dtype=np.uint8)
    counts = np.zeros(n_bins, dtype=int)
    if len(v) == 0:
        return z_bins, mask, counts

    bin_idx = np.floor(v / bin_size).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for i in range(n_bins):
        sel = bin_idx == i
        c = int(sel.sum())
        counts[i] = c
        if c >= min_pts_per_bin:
            z_bins[i] = float(np.median(z[sel]))
            mask[i] = 1
    return z_bins, mask, counts


def interpolate_internal_nans(z_bins: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    z_out = np.array(z_bins, dtype=float, copy=True)
    gap_flag = np.zeros_like(mask, dtype=np.uint8)
    pad_flag = np.zeros_like(mask, dtype=np.uint8)
    valid_idx = np.flatnonzero(mask == 1)
    if len(valid_idx) == 0:
        pad_flag[:] = 1
        return np.zeros_like(z_out), -1, gap_flag, pad_flag

    last_valid_bin = int(valid_idx[-1])
    pad_flag[last_valid_bin + 1 :] = 1
    if len(valid_idx) == 1:
        z_out[: last_valid_bin + 1] = 0.0
        z_out[last_valid_bin] = float(z_bins[last_valid_bin])
        z_out[last_valid_bin + 1 :] = 0.0
        return z_out, last_valid_bin, gap_flag, pad_flag

    x = np.arange(last_valid_bin + 1, dtype=float)
    z_out[: last_valid_bin + 1] = np.interp(x, valid_idx.astype(float), z_bins[valid_idx])
    gap_flag[: last_valid_bin + 1] = (mask[: last_valid_bin + 1] == 0).astype(np.uint8)
    z_out[last_valid_bin + 1 :] = 0.0
    return z_out, last_valid_bin, gap_flag, pad_flag


def smooth_until_last_valid(
    z_interp: np.ndarray,
    last_valid_bin: int,
    sg_window: int = PROFILE_SG_WINDOW,
    sg_poly: int = PROFILE_SG_POLY,
) -> np.ndarray:
    z_s = np.array(z_interp, dtype=float, copy=True)
    if last_valid_bin < 0:
        z_s[:] = 0.0
        return z_s

    valid_len = last_valid_bin + 1
    if valid_len >= sg_window and sg_window % 2 == 1 and sg_window > sg_poly:
        z_s[:valid_len] = savgol_filter(z_s[:valid_len], window_length=sg_window, polyorder=sg_poly, mode="interp")
    z_s[last_valid_bin + 1 :] = 0.0
    return z_s


def apply_sidewalk_edge_buffer(
    v_all: np.ndarray,
    z_all: np.ndarray,
    side_edge_v: float,
    buffer_m: float = SIDEWALK_EDGE_BUFFER_M,
) -> tuple[np.ndarray, np.ndarray, dict]:
    buffer_info = {
        "profile_mode": "sidewalk_edge_buffer" if USE_SIDEWALK_EDGE_BUFFER else "full_profile",
        "sidewalk_edge_buffer_m": float(buffer_m),
        "buffer_applied": 0,
        "buffer_start_v": np.nan,
        "buffer_end_v": np.nan,
        "buffer_width_m": np.nan,
        "sidewalk_edge_center_v": np.nan,
        "sidewalk_edge_center_bin": np.nan,
    }

    if not USE_SIDEWALK_EDGE_BUFFER:
        return v_all, z_all, buffer_info
    if pd.isna(side_edge_v):
        return v_all, z_all, buffer_info

    side_edge_abs = abs(float(side_edge_v))
    buffer_start_v = max(0.0, side_edge_abs - float(buffer_m))
    buffer_end_v = side_edge_abs + float(buffer_m)
    buffer_width_m = buffer_end_v - buffer_start_v

    buffer_info.update({
        "buffer_start_v": float(buffer_start_v),
        "buffer_end_v": float(buffer_end_v),
        "buffer_width_m": float(buffer_width_m),
        "sidewalk_edge_center_v": float(side_edge_abs - buffer_start_v),
        "sidewalk_edge_center_bin": int(np.floor((side_edge_abs - buffer_start_v) / PROFILE_BIN_SIZE)),
    })

    keep_buffer = (v_all >= buffer_start_v) & (v_all <= buffer_end_v)
    if not np.any(keep_buffer):
        return v_all, z_all, buffer_info

    buffer_info["buffer_applied"] = 1
    return v_all[keep_buffer] - buffer_start_v, z_all[keep_buffer], buffer_info


def build_prediction_sample(group_df: pd.DataFrame, lane_groups: dict, source_file: str) -> tuple[dict | None, str]:
    s_key = float(mode_or_first(group_df["_s_key"]))
    s_val = float(mode_or_first(group_df["s_m"]))
    side = str(mode_or_first(group_df["side"]))
    original_label = mode_or_first(group_df["label"]) if "label" in group_df.columns else None
    original_class = mode_or_first(group_df["class"]) if "class" in group_df.columns else None
    original_class = int(original_class) if original_class is not None and pd.notna(original_class) else np.nan

    sw_v_abs, sw_z = prepare_profile_arrays(group_df["v_m"].to_numpy(), group_df["z"].to_numpy())
    lane_df = lane_groups.get((s_key, side), None)
    if lane_df is None:
        lane_v_abs = np.array([], dtype=float)
        lane_z = np.array([], dtype=float)
    else:
        lane_v_abs, lane_z = prepare_profile_arrays(lane_df["v_m"].to_numpy(), lane_df["z"].to_numpy())

    if len(sw_v_abs) == 0 and len(lane_v_abs) == 0:
        return None, "dropped_zero_valid"

    v_all = np.concatenate([sw_v_abs, lane_v_abs]) if len(lane_v_abs) > 0 else sw_v_abs.copy()
    z_all = np.concatenate([sw_z, lane_z]) if len(lane_z) > 0 else sw_z.copy()

    meta, dbg = classify_side(lane_v_abs, lane_z, sw_v_abs, sw_z)
    dbg = dbg or {}
    lane_edge_v = dbg.get("lane_edge_v", np.nan)
    side_edge_v = dbg.get("side_edge_v", np.nan)
    interface_center_v = dbg.get("interface_center_v", np.nan)
    v_all, z_all, buffer_info = apply_sidewalk_edge_buffer(v_all, z_all, side_edge_v)

    z_bins, mask, _ = bin_profile(v_all, z_all)
    z_interp, last_valid_bin, gap_flag, pad_flag = interpolate_internal_nans(z_bins, mask)
    n_valid_bins = int(mask.sum())
    if n_valid_bins == 0:
        return None, "dropped_zero_valid"
    z_s = smooth_until_last_valid(z_interp, last_valid_bin)

    kink_dz = meta.get("dz_m", np.nan)
    kink_slope = meta.get("slope", np.nan)

    row = {f"z_{i:03d}": float(z_s[i]) for i in range(N_BINS)}
    row.update({f"gap_{i:03d}": int(gap_flag[i]) for i in range(N_BINS)})
    row.update({f"pad_{i:03d}": int(pad_flag[i]) for i in range(N_BINS)})
    row.update({
        "s_m": s_val,
        "s_key": s_key,
        "side": side,
        "source_file": source_file,
        "original_label": original_label,
        "original_class": original_class,
        "feat_kink_dz": float(abs(kink_dz)) if pd.notna(kink_dz) else np.nan,
        "feat_kink_slope": float(abs(kink_slope)) if pd.notna(kink_slope) else np.nan,
        **buffer_info,
        "lane_edge_v": float(lane_edge_v) if pd.notna(lane_edge_v) else np.nan,
        "side_edge_v": float(side_edge_v) if pd.notna(side_edge_v) else np.nan,
        "interface_center_v": float(interface_center_v) if pd.notna(interface_center_v) else np.nan,
        "feature_rule_label": meta.get("class", "UNKNOWN"),
        "feature_rule_class": CLASS_CODE.get(meta.get("class", "UNKNOWN"), CLASS_CODE["UNKNOWN"]),
        "n_valid_bins": n_valid_bins,
        "last_valid_bin": int(last_valid_bin),
        "n_sidewalk_pts": int(len(sw_v_abs)),
        "n_lane_pts": int(len(lane_v_abs)),
    })
    return row, "kept"


def build_prediction_dataset(df_sw: pd.DataFrame, df_lane: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, dict]:
    df_sw_keyed = add_station_side_keys(df_sw)
    df_lane_keyed = add_station_side_keys(df_lane)
    lane_groups = {key: grp.copy() for key, grp in df_lane_keyed.groupby(["_s_key", "side"], sort=False)}

    samples = []
    stats = {"total_groups": 0, "kept": 0, "dropped_zero_valid": 0}
    for _, group_df in df_sw_keyed.groupby(["_s_key", "side"], sort=True, dropna=False):
        stats["total_groups"] += 1
        row, status = build_prediction_sample(group_df, lane_groups, source_file)
        stats[status] = stats.get(status, 0) + 1
        if row is not None:
            samples.append(row)

    ordered_cols = PROFILE_COLS + GAP_COLS + PAD_COLS + [
        "feat_kink_dz",
        "feat_kink_slope",
        "profile_mode",
        "sidewalk_edge_buffer_m",
        "buffer_applied",
        "buffer_start_v",
        "buffer_end_v",
        "buffer_width_m",
        "sidewalk_edge_center_v",
        "sidewalk_edge_center_bin",
        "lane_edge_v",
        "side_edge_v",
        "interface_center_v",
        "s_m",
        "s_key",
        "side",
        "source_file",
        "original_label",
        "original_class",
        "feature_rule_label",
        "feature_rule_class",
        "n_valid_bins",
        "last_valid_bin",
        "n_sidewalk_pts",
        "n_lane_pts",
    ]
    df_pred = pd.DataFrame(samples)
    if not df_pred.empty:
        df_pred = df_pred[ordered_cols]
    return df_pred, stats


# -----------------------------------------------------------------------------
# Sidewalk prediction output helpers.
# -----------------------------------------------------------------------------

def apply_predictions_to_sidewalk(df_sw: pd.DataFrame, df_station_predictions: pd.DataFrame) -> pd.DataFrame:
    df_out = add_station_side_keys(df_sw)
    df_out["s_key"] = df_out["_s_key"]
    pred_cols = [
        "s_key",
        "side",
        "pred_class",
        "pred_label",
        "pred_confidence",
        "prob_RAMP",
        "prob_CURB_NO_RAMP",
        "prob_DEPRESSED_DITCH",
    ]
    pred_lookup = df_station_predictions[pred_cols].copy()
    df_out = df_out.merge(pred_lookup, on=["s_key", "side"], how="left")

    rgb = df_out["pred_label"].map(lambda label: CLASS_COLOR.get(label, CLASS_COLOR["UNKNOWN"]))
    df_out["pred_R"] = rgb.map(lambda c: int(c[0]))
    df_out["pred_G"] = rgb.map(lambda c: int(c[1]))
    df_out["pred_B"] = rgb.map(lambda c: int(c[2]))
    df_out["pred_color"] = rgb.map(lambda c: rgb_to_hex(*c))
    return df_out.drop(columns=["_s_key", "s_key"], errors="ignore")


def write_sidewalk_prediction_csv(df_sw_pred: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df_sw_pred.to_csv(path, index=False)
    return path


def write_sidewalk_prediction_asc(df_sw_pred: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df_cc = df_sw_pred[["x", "y", "z", "pred_R", "pred_G", "pred_B", "s_m", "v_m", "side", "pred_class", "pred_confidence"]].copy()
    df_cc["side"] = df_cc["side"].map(SIDE_TO_INT).fillna(df_cc["side"])
    df_cc = df_cc.rename(columns={"x": "//X"})
    with path.open("w", encoding="utf-8") as f:
        f.write("//X Y Z R G B s_m v_m side pred_class pred_confidence\n")
        df_cc.to_csv(f, sep=" ", index=False, header=False)
    return path


def write_sidewalk_prediction_las(df_sw_pred: pd.DataFrame, path: str | Path) -> Path:
    import laspy

    path = Path(path)
    header = laspy.LasHeader(point_format=2, version="1.4")
    las = laspy.LasData(header=header)
    las.x = df_sw_pred["x"].to_numpy(dtype=float)
    las.y = df_sw_pred["y"].to_numpy(dtype=float)
    las.z = df_sw_pred["z"].to_numpy(dtype=float)
    las.red = df_sw_pred["pred_R"].fillna(CLASS_COLOR["UNKNOWN"][0]).to_numpy(dtype=np.uint16) * 256
    las.green = df_sw_pred["pred_G"].fillna(CLASS_COLOR["UNKNOWN"][1]).to_numpy(dtype=np.uint16) * 256
    las.blue = df_sw_pred["pred_B"].fillna(CLASS_COLOR["UNKNOWN"][2]).to_numpy(dtype=np.uint16) * 256
    las.write(path)
    return path
