"""
Stage 5 — Generate YOLO label files from RSNA 2024 coordinate annotations.

Each output label file corresponds to one preprocessed PNG image produced by
Stage 4 (pad_and_resize).  Lines follow the YOLO format:

    <class_id> <x_center> <y_center> <width> <height>

where all box coordinates are normalised to [0, 1] relative to a 640×640
canvas.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.bbox_utils import pixel_to_yolo, get_class_id as _bbox_get_class_id

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SEVERITY_MAP: dict[str, int] = {
    "normal/mild": 0,
    "moderate":    1,
    "severe":      2,
}

CONDITION_BASE: dict[str, int] = {
    "spinal canal stenosis":             0,
    "right neural foraminal narrowing":  3,
    "left neural foraminal narrowing":   6,
    "right subarticular stenosis":       9,
    "left subarticular stenosis":        12,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_str(s: str) -> str:
    """Lowercase and strip a string."""
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_merge(raw_root: Path, manifest_path: Path) -> pd.DataFrame:
    """
    Load and merge the three source tables into a single annotation DataFrame.

    Parameters
    ----------
    raw_root : Path
        Root directory that contains ``train.csv`` and
        ``train_label_coordinates.csv``.
    manifest_path : Path
        Path to ``data/processed/manifest.csv`` produced by Stage 4.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns from all three source tables.  Each row
        represents one keypoint annotation for one image slice.

    Notes
    -----
    String columns ``condition``, ``level``, and ``severity`` are normalised
    (lowercase + stripped) before any merge.  ``series_id`` and
    ``instance_number`` are cast to ``str`` to guarantee key-type alignment
    between the coordinate table and the manifest.
    """
    # --- load sources -------------------------------------------------------
    label_df = pd.read_csv(raw_root / "train.csv")
    coord_df = pd.read_csv(raw_root / "train_label_coordinates.csv")
    manifest_df = pd.read_csv(manifest_path)

    # --- normalise string columns -------------------------------------------
    for df, cols in [
        (label_df,    ["condition", "level", "severity"]),
        (coord_df,    ["condition", "level"]),
        (manifest_df, []),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

    # severity only in label_df
    if "severity" in label_df.columns:
        label_df["severity"] = label_df["severity"].astype(str).str.strip().str.lower()

    # --- merge coord_df with label_df on (study_id, condition, level) -------
    before = len(coord_df)
    merged_df = coord_df.merge(
        label_df,
        on=["study_id", "condition", "level"],
        how="inner",
    )
    after = len(merged_df)
    logger.info("coord+label merge: %d rows → %d rows", before, after)
    if after < before:
        logger.warning(
            "coord+label merge dropped %d rows (unmatched study/condition/level)",
            before - after,
        )

    # --- cast join keys to str before merging with manifest -----------------
    for col in ["series_id", "instance_number"]:
        merged_df[col] = merged_df[col].astype(str)
        manifest_df[col] = manifest_df[col].astype(str)

    # cast study_id to same type
    merged_df["study_id"] = merged_df["study_id"].astype(str)
    manifest_df["study_id"] = manifest_df["study_id"].astype(str)

    # --- merge with manifest on (study_id, series_id, instance_number) ------
    before2 = len(merged_df)
    final_df = merged_df.merge(
        manifest_df,
        on=["study_id", "series_id", "instance_number"],
        how="inner",
    )
    logger.info(
        "annotation+manifest merge: %d rows → %d rows", before2, len(final_df)
    )

    return final_df


def build_class_id(condition: str, severity: str) -> int:
    """
    Compute the YOLO class ID from a condition and severity string.

    Parameters
    ----------
    condition : str
        Condition name in space-separated format, e.g.
        ``"spinal canal stenosis"``.  Case-insensitive; whitespace stripped.
    severity : str
        Severity label, e.g. ``"normal/mild"``, ``"moderate"``, or
        ``"severe"``.  Case-insensitive; whitespace stripped.

    Returns
    -------
    int
        Class ID in [0, 14].

    Raises
    ------
    ValueError
        If ``condition`` is not in ``CONDITION_BASE`` or ``severity`` is not
        in ``SEVERITY_MAP``.
    """
    cond_key = _norm_str(condition)
    sev_key  = _norm_str(severity)

    if cond_key not in CONDITION_BASE:
        raise ValueError(
            f"Unknown condition {condition!r}. "
            f"Valid options: {sorted(CONDITION_BASE)}"
        )
    if sev_key not in SEVERITY_MAP:
        raise ValueError(
            f"Unknown severity {severity!r}. "
            f"Valid options: {sorted(SEVERITY_MAP)}"
        )

    return CONDITION_BASE[cond_key] + SEVERITY_MAP[sev_key]


def make_label_for_image(
    group: pd.DataFrame,
    output_root: Path,
    target_size: int = 640,
) -> int:
    """
    Write a YOLO label file for all annotations belonging to one image.

    Parameters
    ----------
    group : pd.DataFrame
        Subset of the merged DataFrame for a single image (same
        ``output_path``).  Must contain columns: ``output_path``,
        ``series_type``, ``original_rows``, ``original_cols``, ``x``, ``y``,
        ``condition``, ``severity``.
    output_root : Path
        Root directory under which ``labels/{series_type}/{stem}.txt`` is
        written.
    target_size : int, optional
        Side length of the square canvas used by pad_and_resize (default 640).

    Returns
    -------
    int
        Number of annotation lines written to the label file.
    """
    lines: list[str] = []

    for _, row in group.iterrows():
        try:
            original_rows = int(row["original_rows"])
            original_cols = int(row["original_cols"])
            condition_raw  = _norm_str(row["condition"])   # space-format
            severity_raw   = _norm_str(row["severity"])

            # --- replicate pad_and_resize coordinate transform --------------
            scale   = target_size / max(original_rows, original_cols)
            new_h   = int(original_rows * scale)
            new_w   = int(original_cols * scale)
            pad_top  = (target_size - new_h) // 2
            pad_left = (target_size - new_w) // 2

            x_scaled = float(row["x"]) * scale + pad_left
            y_scaled = float(row["y"]) * scale + pad_top

            # --- clamp with warning -----------------------------------------
            if not (0.0 <= x_scaled <= target_size):
                logger.warning(
                    "x_scaled=%.2f out of [0, %d] for %s; clamping",
                    x_scaled, target_size, row.get("output_path", "?"),
                )
                x_scaled = max(0.0, min(float(target_size), x_scaled))

            if not (0.0 <= y_scaled <= target_size):
                logger.warning(
                    "y_scaled=%.2f out of [0, %d] for %s; clamping",
                    y_scaled, target_size, row.get("output_path", "?"),
                )
                y_scaled = max(0.0, min(float(target_size), y_scaled))

            # --- convert to YOLO format -------------------------------------
            # pixel_to_yolo expects underscore-format condition
            condition_under = condition_raw.replace(" ", "_")
            x_norm, y_norm, w_norm, h_norm = pixel_to_yolo(
                x_scaled, y_scaled, target_size, target_size, condition_under
            )

            # --- class ID ---------------------------------------------------
            class_id = build_class_id(condition_raw, severity_raw)

            lines.append(
                f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping annotation row due to error: %s", exc, exc_info=False
            )

    # --- determine label file path ------------------------------------------
    first_row  = group.iloc[0]
    output_path = Path(str(first_row["output_path"]))
    series_type = _norm_str(first_row["series_type"])
    stem        = output_path.stem

    label_dir  = output_root / "labels" / series_type
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{stem}.txt"

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return len(lines)


def make_all_labels(
    raw_root: Path,
    manifest_path: Path,
    output_root: Path,
) -> pd.DataFrame:
    """
    Generate YOLO label files for every image listed in the manifest.

    Images that have no matching annotations receive an empty ``.txt`` file,
    as required by YOLO training pipelines.

    Parameters
    ----------
    raw_root : Path
        Directory containing the raw competition CSVs.
    manifest_path : Path
        Path to ``data/processed/manifest.csv``.
    output_root : Path
        Root output directory.  Label files are written to
        ``output_root/labels/{series_type}/{stem}.txt``.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per image and columns:
        ``output_path``, ``label_path``, ``n_annotations``, ``series_type``.
    """
    merged_df   = load_and_merge(raw_root, manifest_path)
    manifest_df = pd.read_csv(manifest_path)

    # Cast key columns to str to match merged_df
    for col in ["series_id", "instance_number", "study_id"]:
        manifest_df[col] = manifest_df[col].astype(str)

    # Build a lookup: output_path → sub-DataFrame of annotations
    annotated: dict[str, pd.DataFrame] = dict(
        tuple(merged_df.groupby("output_path"))  # type: ignore[arg-type]
    )

    summary_rows: list[dict] = []
    total_annotations = 0

    for _, manifest_row in manifest_df.iterrows():
        op          = str(manifest_row["output_path"])
        series_type = _norm_str(manifest_row["series_type"])
        stem        = Path(op).stem
        label_dir   = output_root / "labels" / series_type
        label_path  = label_dir / f"{stem}.txt"

        try:
            if op in annotated:
                n = make_label_for_image(annotated[op], output_root)
            else:
                # Empty label file — required by YOLO
                label_dir.mkdir(parents=True, exist_ok=True)
                label_path.write_text("")
                n = 0

            total_annotations += n
            summary_rows.append(
                {
                    "output_path":   op,
                    "label_path":    str(label_path),
                    "n_annotations": n,
                    "series_type":   series_type,
                }
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process %s: %s", op, exc, exc_info=False)
            summary_rows.append(
                {
                    "output_path":   op,
                    "label_path":    str(label_path),
                    "n_annotations": 0,
                    "series_type":   series_type,
                }
            )

    logger.info("Total annotations written: %d", total_annotations)
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface for Stage 5 label generation.

    Parameters
    ----------
    None — arguments are read from ``sys.argv`` via :mod:`argparse`.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Stage 5: Generate YOLO label files for the RSNA 2024 spine dataset."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Directory containing train.csv and train_label_coordinates.csv",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Path to manifest.csv (default: data/processed/manifest.csv)",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed",
        help="Root output directory (default: data/processed)",
    )
    args = parser.parse_args()

    raw_root    = Path(args.raw_root)
    manifest    = Path(args.manifest)
    output_root = Path(args.output_root)

    summary_df = make_all_labels(raw_root, manifest, output_root)

    # Save summary
    summary_path = output_root / "labels_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Labels summary saved to %s", summary_path)

    # Print statistics
    total = int(summary_df["n_annotations"].sum())
    print(f"\nTotal annotations: {total}")
    print("\nPer-series-type breakdown:")
    breakdown = (
        summary_df.groupby("series_type")["n_annotations"]
        .agg(images="count", annotations="sum")
        .reset_index()
    )
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
