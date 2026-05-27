"""Apply sparse manual station-side class corrections.

This script is the standalone version of the remapping cells from
``updated_ditch_latest_version.ipynb``. It is intentionally separate from the
core detection pipeline because manual correction is a post-processing step:
the rule-based pipeline writes its original labels first, then this script can
create a sparse correction template or apply edited corrections later.

Correction CSV format
---------------------
The corrector CSV has exactly the three important columns below. Add only rows
that need to change; every station-side not listed is left untouched.

``s_m``
    Station value, rounded to 3 decimals for matching.

``side``
    ``RIGHT`` or ``LEFT``.

``class``
    Integer class code from the Stage 1 class map.

Typical workflow
----------------
1. Create a template once:

   ``python remap_station_side_classes.py --base path/to/traj --create-template``

2. Edit ``path/to/traj_station_side_class_corrector.csv`` manually.

3. Apply the corrections:

   ``python remap_station_side_classes.py --base path/to/traj --apply``

Using ``--base`` follows the pipeline naming convention. For example,
``--base 02/segments/part2/traj`` derives ``traj_sidewalk.csv``,
``traj_traj.csv``, and ``traj_station_side_class_corrector.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
PRIORITY = {name: i for i, name in enumerate(CLASS_CODE)}
SIDE_TO_INT = {"RIGHT": 0, "LEFT": 1}


@dataclass
class RemapPaths:
    """Input and output paths used by the remapping script.

    Sidewalk and trajectory remapping are independent. A caller may provide only
    a sidewalk CSV, only a trajectory CSV, or both. The correction CSV is shared
    because both products are keyed by rounded ``s_m`` and side.
    """

    corrector_csv: Path
    sidewalk_csv: Path | None = None
    trajectory_csv: Path | None = None
    sidewalk_out_csv: Path | None = None
    sidewalk_out_asc: Path | None = None
    trajectory_out_csv: Path | None = None
    trajectory_out_asc: Path | None = None


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Return a CSS-style hex color for 0-255 RGB values."""

    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def dominant_class(class_right: str, class_left: str) -> str:
    """Pick the higher-priority station class for trajectory coloring."""

    if PRIORITY.get(class_right, 99) <= PRIORITY.get(class_left, 99):
        return class_right
    return class_left


def derive_paths_from_base(base: Path) -> RemapPaths:
    """Derive standard Stage 1 remapping paths from an output base prefix.

    The core pipeline writes files like ``<base>_sidewalk.csv`` and
    ``<base>_traj.csv``. This helper keeps CLI usage short while preserving the
    same names used by the notebook.
    """

    return RemapPaths(
        corrector_csv=base.with_name(base.name + "_station_side_class_corrector.csv"),
        sidewalk_csv=base.with_name(base.name + "_sidewalk.csv"),
        trajectory_csv=base.with_name(base.name + "_traj.csv"),
        sidewalk_out_csv=base.with_name(base.name + "_sidewalk_remapped.csv"),
        sidewalk_out_asc=base.with_name(base.name + "_sidewalk_remapped_CC.asc"),
        trajectory_out_csv=base.with_name(base.name + "_traj_remapped.csv"),
        trajectory_out_asc=base.with_name(base.name + "_traj_remapped_CC.asc"),
    )


def default_sidewalk_outputs(sidewalk_csv: Path) -> tuple[Path, Path]:
    """Return default remapped sidewalk CSV/ASC names for an explicit input."""

    stem = sidewalk_csv.stem
    if stem.endswith("_sidewalk"):
        base = stem[: -len("_sidewalk")]
        return sidewalk_csv.with_name(base + "_sidewalk_remapped.csv"), sidewalk_csv.with_name(base + "_sidewalk_remapped_CC.asc")
    return sidewalk_csv.with_name(stem + "_remapped.csv"), sidewalk_csv.with_name(stem + "_remapped_CC.asc")


def default_trajectory_outputs(trajectory_csv: Path) -> tuple[Path, Path]:
    """Return default remapped trajectory CSV/ASC names for an explicit input."""

    stem = trajectory_csv.stem
    if stem.endswith("_traj"):
        base = stem[: -len("_traj")]
        return trajectory_csv.with_name(base + "_traj_remapped.csv"), trajectory_csv.with_name(base + "_traj_remapped_CC.asc")
    return trajectory_csv.with_name(stem + "_remapped.csv"), trajectory_csv.with_name(stem + "_remapped_CC.asc")


def create_corrector_template(path: Path, overwrite: bool = False) -> None:
    """Create an empty sparse correction CSV template.

    Existing templates are left untouched by default so manual edits are not
    accidentally overwritten. Use ``--overwrite-template`` only when you really
    want to reset the correction file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"Sparse station-side corrector CSV already exists -> {path}")
        print("Leaving the existing file untouched so manual edits are not overwritten.")
        return
    pd.DataFrame(columns=["s_m", "side", "class"]).to_csv(path, index=False)
    print(f"Sparse station-side corrector CSV template saved -> {path}")
    print("Add one row per station-side change only, for example: s_m=12.345, side=RIGHT, class=2")


def load_corrector(path: Path) -> pd.DataFrame:
    """Load, validate, normalize, and de-duplicate correction rows.

    Corrections match exported products by ``round(s_m, 3)`` and normalized side
    text. If duplicate rows target the same station-side, the last row wins,
    matching the notebook behavior and making manual CSV edits predictable.
    """

    if not path.exists():
        raise FileNotFoundError(f"Correction CSV does not exist: {path}")
    df_map = pd.read_csv(path)
    required_cols = {"s_m", "side", "class"}
    if not required_cols.issubset(df_map.columns):
        raise ValueError(f"Correction CSV must contain columns: {sorted(required_cols)}")

    df_map = df_map.copy()
    df_map = df_map.dropna(how="all")
    df_map = df_map[["s_m", "side", "class"]].dropna(subset=["s_m", "side", "class"])
    df_map["_s_key"] = df_map["s_m"].astype(float).round(3)
    df_map["_side_key"] = df_map["side"].astype(str).str.upper().str.strip()
    df_map["class"] = df_map["class"].astype(int)

    bad_sides = sorted(set(df_map["_side_key"]) - set(SIDE_TO_INT))
    if bad_sides:
        raise ValueError(f"Unknown side values in correction CSV: {bad_sides}. Use RIGHT or LEFT.")

    bad_codes = sorted(set(df_map["class"]) - set(INV_MAP))
    if bad_codes:
        raise ValueError(f"Unknown class codes in correction CSV: {bad_codes}")

    dupes = df_map.duplicated(subset=["_s_key", "_side_key"], keep=False)
    if dupes.any():
        n_dupe_rows = int(dupes.sum())
        n_dupe_keys = int(df_map.loc[dupes, ["_s_key", "_side_key"]].drop_duplicates().shape[0])
        print(f"Found {n_dupe_rows} duplicate correction rows across {n_dupe_keys} (s_m, side) keys; keeping the last row for each key.")
        df_map = df_map.drop_duplicates(subset=["_s_key", "_side_key"], keep="last")

    return df_map


def apply_sidewalk_remap(df_sw: pd.DataFrame, df_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Apply station-side class corrections to a sidewalk dataframe.

    Only class-related columns are changed. Geometry columns, station values,
    cluster IDs, and point order are preserved. The returned boolean ``matched``
    has one element per sidewalk row and marks rows updated by the correction
    file.
    """

    required_cols = {"s_m", "side", "class"}
    if not required_cols.issubset(df_sw.columns):
        raise ValueError(f"Sidewalk CSV must contain columns: {sorted(required_cols)}")

    original_columns = list(df_sw.columns)
    original_count = len(df_sw)

    df_out = df_sw.copy()
    df_out["_s_key"] = df_out["s_m"].astype(float).round(3)
    df_out["_side_key"] = df_out["side"].astype(str).str.upper().str.strip()

    # Left merge marks only rows whose station-side appears in the sparse map.
    df_out = df_out.merge(
        df_map[["_s_key", "_side_key", "class"]].rename(columns={"class": "class_new"}),
        on=["_s_key", "_side_key"],
        how="left",
    )
    matched = df_out["class_new"].notna()
    df_out.loc[matched, "class"] = df_out.loc[matched, "class_new"].astype(int)

    if matched.any():
        new_label = df_out.loc[matched, "class"].astype(int).map(INV_MAP)

        if "label" in df_out.columns:
            df_out.loc[matched, "label"] = new_label

        if {"R", "G", "B"}.issubset(df_out.columns):
            rgb = new_label.map(CLASS_COLOR)
            df_out.loc[matched, "R"] = rgb.map(lambda c: int(c[0]))
            df_out.loc[matched, "G"] = rgb.map(lambda c: int(c[1]))
            df_out.loc[matched, "B"] = rgb.map(lambda c: int(c[2]))

        if "color" in df_out.columns and {"R", "G", "B"}.issubset(df_out.columns):
            df_out.loc[matched, "color"] = df_out.loc[matched].apply(
                lambda row: rgb_to_hex(int(row["R"]), int(row["G"]), int(row["B"])), axis=1
            )

        if "cluster_label" in df_out.columns:
            df_out.loc[matched, "cluster_label"] = new_label

    unmatched = df_map.merge(
        df_out[["_s_key", "_side_key"]].drop_duplicates(),
        on=["_s_key", "_side_key"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]

    df_out = df_out.drop(columns=["_s_key", "_side_key", "class_new"])
    df_out = df_out[original_columns]
    assert len(df_out) == original_count
    return df_out, matched, unmatched


def write_sidewalk_asc(df_sw: pd.DataFrame, path: Path) -> None:
    """Write a CloudCompare-style ASC for remapped sidewalk points."""

    cc_cols = ["x", "y", "z", "R", "G", "B", "s_m", "v_m", "side", "class"]
    for optional_col in ["cluster_id", "cluster_role_code", "ambiguous"]:
        if optional_col in df_sw.columns:
            cc_cols.append(optional_col)
    missing = [c for c in cc_cols if c not in df_sw.columns]
    if missing:
        raise ValueError(f"Sidewalk CSV missing CloudCompare columns: {missing}")

    df_cc = df_sw[cc_cols].copy()
    df_cc["side"] = df_cc["side"].map(SIDE_TO_INT).fillna(df_cc["side"])
    df_cc = df_cc.rename(columns={"x": "//X"})

    header_cols = ["//X" if c == "x" else c for c in cc_cols]
    header_cols = ["cluster_role" if c == "cluster_role_code" else c for c in header_cols]
    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(header_cols) + "\n")
        df_cc.to_csv(f, sep=" ", index=False, header=False)


def remap_sidewalk(sidewalk_csv: Path, corrector_csv: Path, output_csv: Path, output_asc: Path) -> None:
    """Read sidewalk/correction CSVs, apply corrections, and write CSV/ASC."""

    df_map = load_corrector(corrector_csv)
    df_sw = pd.read_csv(sidewalk_csv)
    df_out, matched, unmatched = apply_sidewalk_remap(df_sw, df_map)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    write_sidewalk_asc(df_out, output_asc)

    print(f"Remapped sidewalk CSV saved -> {output_csv}")
    print(f"Remapped sidewalk ASC saved -> {output_asc}")
    print(f"Correction rows read: {len(df_map)}")
    print(f"Sidewalk rows updated from correction CSV: {int(matched.sum())} / {len(df_out)}")
    if len(unmatched) > 0:
        print("Correction rows with no matching sidewalk points:")
        print(unmatched[["s_m", "side", "class"]].head(20))


def apply_trajectory_remap(df_traj: pd.DataFrame, df_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Apply station-side corrections to a trajectory dataframe.

    Trajectory rows contain both side classes. A RIGHT correction updates
    ``class_RIGHT`` and optional ``label_R``; a LEFT correction updates
    ``class_LEFT`` and optional ``label_L``. After side updates, dominant class
    and RGB columns are recomputed from the current RIGHT/LEFT labels.
    """

    required_cols = {"s_m", "class_RIGHT", "class_LEFT", "R", "G", "B"}
    if not required_cols.issubset(df_traj.columns):
        raise ValueError(f"Trajectory CSV must contain columns: {sorted(required_cols)}")

    original_count = len(df_traj)
    df_out = df_traj.copy()
    df_out["_s_key"] = df_out["s_m"].astype(float).round(3)
    new_cols = []

    # Apply RIGHT and LEFT corrections separately because both side labels live
    # in the same trajectory row.
    for side_key, class_col, label_col in [("RIGHT", "class_RIGHT", "label_R"), ("LEFT", "class_LEFT", "label_L")]:
        new_col = f"{class_col}_new"
        side_map = df_map[df_map["_side_key"] == side_key][["_s_key", "class"]]
        side_map = side_map.rename(columns={"class": new_col})
        df_out = df_out.merge(side_map, on="_s_key", how="left")
        matched = df_out[new_col].notna()
        df_out.loc[matched, class_col] = df_out.loc[matched, new_col].astype(int)
        if label_col in df_out.columns:
            df_out.loc[matched, label_col] = df_out.loc[matched, class_col].astype(int).map(INV_MAP)
        new_cols.append(new_col)

    label_r = df_out["class_RIGHT"].astype(int).map(INV_MAP).fillna("UNKNOWN")
    label_l = df_out["class_LEFT"].astype(int).map(INV_MAP).fillna("UNKNOWN")
    dom_labels = [dominant_class(r, l) for r, l in zip(label_r, label_l)]
    dom_rgb = [CLASS_COLOR.get(label, CLASS_COLOR["UNKNOWN"]) for label in dom_labels]

    df_out["R"] = [int(rgb[0]) for rgb in dom_rgb]
    df_out["G"] = [int(rgb[1]) for rgb in dom_rgb]
    df_out["B"] = [int(rgb[2]) for rgb in dom_rgb]
    if "dominant" in df_out.columns:
        df_out["dominant"] = dom_labels
    if "hex_R" in df_out.columns:
        df_out["hex_R"] = label_r.map(lambda label: rgb_to_hex(*CLASS_COLOR.get(label, CLASS_COLOR["UNKNOWN"])))
    if "hex_L" in df_out.columns:
        df_out["hex_L"] = label_l.map(lambda label: rgb_to_hex(*CLASS_COLOR.get(label, CLASS_COLOR["UNKNOWN"])))

    matched_any = df_out[new_cols].notna().any(axis=1)
    unmatched = df_map.merge(
        df_out[["_s_key"]].drop_duplicates(),
        on="_s_key",
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]

    df_out = df_out.drop(columns=["_s_key"] + new_cols)
    assert len(df_out) == original_count
    return df_out, matched_any, unmatched


def write_trajectory_asc(df_traj: pd.DataFrame, path: Path) -> None:
    """Write a CloudCompare-style ASC for remapped trajectory points."""

    cc_cols = ["//X", "Y", "Z", "R", "G", "B", "s_m", "class_RIGHT", "class_LEFT", "ambiguous_RIGHT", "ambiguous_LEFT"]
    missing = [c for c in cc_cols if c not in df_traj.columns]
    if missing:
        raise ValueError(f"Trajectory CSV missing CloudCompare columns: {missing}")
    with path.open("w", encoding="utf-8") as f:
        f.write("//X Y Z R G B s_m class_RIGHT class_LEFT ambiguous_RIGHT ambiguous_LEFT\n")
        df_traj[cc_cols].to_csv(f, sep=" ", index=False, header=False)


def remap_trajectory(trajectory_csv: Path, corrector_csv: Path, output_csv: Path, output_asc: Path) -> None:
    """Read trajectory/correction CSVs, apply corrections, and write CSV/ASC."""

    df_map = load_corrector(corrector_csv)
    df_traj = pd.read_csv(trajectory_csv)
    df_out, matched_any, unmatched = apply_trajectory_remap(df_traj, df_map)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    write_trajectory_asc(df_out, output_asc)

    print(f"Remapped trajectory CSV saved -> {output_csv}")
    print(f"Remapped trajectory ASC saved -> {output_asc}")
    print(f"Correction rows read: {len(df_map)}")
    print(f"Trajectory rows updated from correction CSV: {int(matched_any.sum())} / {len(df_out)}")
    if len(unmatched) > 0:
        print("Correction rows with no matching trajectory station:")
        print(unmatched[["s_m", "side", "class"]].head(20))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for correction template/remapping tasks."""

    parser = argparse.ArgumentParser(description="Create/apply sparse station-side class corrections for Stage 1 outputs.")
    parser.add_argument("--base", type=Path, help="Output base prefix, e.g. 02/segments/part2/traj. Derives standard filenames.")
    parser.add_argument("--corrector-csv", type=Path, help="Sparse correction CSV. Defaults to <base>_station_side_class_corrector.csv.")
    parser.add_argument("--sidewalk-csv", type=Path, help="Input sidewalk CSV to remap. Defaults to <base>_sidewalk.csv.")
    parser.add_argument("--trajectory-csv", type=Path, help="Input trajectory CSV to remap. Defaults to <base>_traj.csv.")
    parser.add_argument("--sidewalk-out-csv", type=Path, help="Output remapped sidewalk CSV.")
    parser.add_argument("--sidewalk-out-asc", type=Path, help="Output remapped sidewalk CloudCompare ASC.")
    parser.add_argument("--trajectory-out-csv", type=Path, help="Output remapped trajectory CSV.")
    parser.add_argument("--trajectory-out-asc", type=Path, help="Output remapped trajectory CloudCompare ASC.")
    parser.add_argument("--create-template", action="store_true", help="Create the sparse corrector CSV template.")
    parser.add_argument("--overwrite-template", action="store_true", help="Overwrite an existing template when used with --create-template.")
    parser.add_argument("--apply", action="store_true", help="Apply corrections to supplied/derived output CSVs.")
    parser.add_argument("--sidewalk-only", action="store_true", help="When applying, remap only the sidewalk CSV.")
    parser.add_argument("--trajectory-only", action="store_true", help="When applying, remap only the trajectory CSV.")
    return parser


def resolve_paths(args: argparse.Namespace) -> RemapPaths:
    """Resolve explicit CLI paths plus optional --base-derived defaults."""

    if args.base is None and args.corrector_csv is None:
        raise ValueError("Provide either --base or --corrector-csv.")

    paths = derive_paths_from_base(args.base) if args.base is not None else RemapPaths(corrector_csv=args.corrector_csv)

    if args.corrector_csv is not None:
        paths.corrector_csv = args.corrector_csv
    if args.sidewalk_csv is not None:
        paths.sidewalk_csv = args.sidewalk_csv
    if args.trajectory_csv is not None:
        paths.trajectory_csv = args.trajectory_csv

    if args.sidewalk_out_csv is not None:
        paths.sidewalk_out_csv = args.sidewalk_out_csv
    if args.sidewalk_out_asc is not None:
        paths.sidewalk_out_asc = args.sidewalk_out_asc
    if args.trajectory_out_csv is not None:
        paths.trajectory_out_csv = args.trajectory_out_csv
    if args.trajectory_out_asc is not None:
        paths.trajectory_out_asc = args.trajectory_out_asc

    if paths.sidewalk_csv is not None and (paths.sidewalk_out_csv is None or paths.sidewalk_out_asc is None):
        default_csv, default_asc = default_sidewalk_outputs(paths.sidewalk_csv)
        paths.sidewalk_out_csv = paths.sidewalk_out_csv or default_csv
        paths.sidewalk_out_asc = paths.sidewalk_out_asc or default_asc

    if paths.trajectory_csv is not None and (paths.trajectory_out_csv is None or paths.trajectory_out_asc is None):
        default_csv, default_asc = default_trajectory_outputs(paths.trajectory_csv)
        paths.trajectory_out_csv = paths.trajectory_out_csv or default_csv
        paths.trajectory_out_asc = paths.trajectory_out_asc or default_asc

    return paths


def validate_apply_request(args: argparse.Namespace, paths: RemapPaths) -> tuple[bool, bool]:
    """Validate apply-mode options and return sidewalk/trajectory booleans."""

    if args.sidewalk_only and args.trajectory_only:
        raise ValueError("Use only one of --sidewalk-only or --trajectory-only.")

    do_sidewalk = not args.trajectory_only
    do_trajectory = not args.sidewalk_only

    if do_sidewalk and paths.sidewalk_csv is None:
        raise ValueError("No sidewalk CSV resolved. Provide --base, --sidewalk-csv, or use --trajectory-only.")
    if do_trajectory and paths.trajectory_csv is None:
        raise ValueError("No trajectory CSV resolved. Provide --base, --trajectory-csv, or use --sidewalk-only.")

    if not paths.corrector_csv.exists():
        raise FileNotFoundError(f"Correction CSV does not exist: {paths.corrector_csv}")
    if do_sidewalk and not paths.sidewalk_csv.exists():
        raise FileNotFoundError(f"Sidewalk CSV does not exist: {paths.sidewalk_csv}")
    if do_trajectory and not paths.trajectory_csv.exists():
        raise FileNotFoundError(f"Trajectory CSV does not exist: {paths.trajectory_csv}")

    return do_sidewalk, do_trajectory


def main() -> None:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    paths = resolve_paths(args)

    if not args.create_template and not args.apply:
        parser.error("Choose at least one action: --create-template and/or --apply.")

    if args.create_template:
        create_corrector_template(paths.corrector_csv, overwrite=args.overwrite_template)

    if args.apply:
        do_sidewalk, do_trajectory = validate_apply_request(args, paths)
        if do_sidewalk:
            remap_sidewalk(paths.sidewalk_csv, paths.corrector_csv, paths.sidewalk_out_csv, paths.sidewalk_out_asc)
        if do_trajectory:
            remap_trajectory(paths.trajectory_csv, paths.corrector_csv, paths.trajectory_out_csv, paths.trajectory_out_asc)


if __name__ == "__main__":
    main()
