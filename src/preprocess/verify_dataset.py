"""
Dataset verification script for the RSNA 2024 Lumbar Spine Degenerative
Classification pipeline.

Acts as a CI gate: exits with code 1 if any critical check fails.

Usage
-----
python -m src.preprocess.verify_dataset \\
    --processed-root data/processed \\
    --splits-dir data/splits \\
    --sample-n 100
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.utils.bbox_utils import CLASS_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_image_label_pairing(processed_root: Path) -> list[str]:
    """
    Verify that every PNG image has a corresponding label file.

    Parameters
    ----------
    processed_root : Path
        Root of the processed data directory, expected to contain
        ``images/`` and ``labels/`` subdirectories.

    Returns
    -------
    list[str]
        Error strings for each PNG whose matching ``.txt`` label file is
        missing.  An empty list means all pairings are valid.

    Notes
    -----
    The label path is derived by replacing the ``images/`` path segment
    with ``labels/`` and swapping the ``.png`` extension for ``.txt``.

    Example::

        images/sagittal_t1/12345_67890_1.png
        → labels/sagittal_t1/12345_67890_1.txt
    """
    images_dir = processed_root / "images"
    labels_dir = processed_root / "labels"
    errors: list[str] = []

    png_files = list(images_dir.rglob("*.png"))
    for png_path in png_files:
        relative = png_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix(".txt")
        if not label_path.exists():
            errors.append(
                f"Missing label file for image: {png_path} "
                f"(expected: {label_path})"
            )
    return errors


def check_label_format(label_path: Path) -> list[str]:
    """
    Validate the YOLO annotation format of a single label file.

    Parameters
    ----------
    label_path : Path
        Path to the ``.txt`` label file to validate.

    Returns
    -------
    list[str]
        One error string per malformed line (includes line number and
        content).  An empty list means the file is valid.  Empty files
        are considered valid.

    Notes
    -----
    Each non-empty line must satisfy all of the following:

    1. Exactly 5 space-separated tokens.
    2. Token 0 (class_id): integer in [0, 14].
    3. Tokens 1-4 (x_center, y_center, width, height): float, each in
       the half-open interval (0.0, 1.0] (strictly > 0 and <= 1).
    """
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8")

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) != 5:
            errors.append(
                f"{label_path}:{lineno}: expected 5 tokens, got "
                f"{len(tokens)}: {line!r}"
            )
            continue

        # Validate class_id (token 0)
        try:
            class_id = int(tokens[0])
        except ValueError:
            errors.append(
                f"{label_path}:{lineno}: class_id is not an integer: {line!r}"
            )
            continue

        if not (0 <= class_id <= 14):
            errors.append(
                f"{label_path}:{lineno}: class_id {class_id} not in [0, 14]: "
                f"{line!r}"
            )
            # Still validate coords even with bad class_id
            # fall through to coord check below

        # Validate coordinates (tokens 1-4)
        coord_error = False
        for idx in range(1, 5):
            try:
                val = float(tokens[idx])
            except ValueError:
                errors.append(
                    f"{label_path}:{lineno}: token {idx} is not a float: "
                    f"{line!r}"
                )
                coord_error = True
                break
            if not (0.0 < val <= 1.0):
                errors.append(
                    f"{label_path}:{lineno}: token {idx} value {val} not in "
                    f"(0.0, 1.0]: {line!r}"
                )
                coord_error = True
                break

    return errors


def check_all_labels(
    processed_root: Path, max_errors_shown: int = 20
) -> dict:
    """
    Run format checks on every label file under ``processed_root/labels/``.

    Parameters
    ----------
    processed_root : Path
        Root of the processed data directory.
    max_errors_shown : int, optional
        Maximum number of error strings to include in the returned dict.
        Defaults to 20.

    Returns
    -------
    dict
        A summary dictionary with the following keys:

        - ``"total_label_files"`` (int): total ``.txt`` files found.
        - ``"files_with_errors"`` (int): count of files with >= 1 error.
        - ``"total_annotation_lines"`` (int): total non-empty annotation
          lines across all files.
        - ``"empty_label_files"`` (int): files with zero non-empty lines.
        - ``"errors"`` (list[str]): error strings capped at
          ``max_errors_shown``.
    """
    labels_dir = processed_root / "labels"
    txt_files = list(labels_dir.rglob("*.txt"))

    total_label_files = len(txt_files)
    files_with_errors = 0
    total_annotation_lines = 0
    empty_label_files = 0
    all_errors: list[str] = []

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8")
        non_empty_lines = [l for l in text.splitlines() if l.strip()]
        total_annotation_lines += len(non_empty_lines)
        if len(non_empty_lines) == 0:
            empty_label_files += 1

        file_errors = check_label_format(txt_path)
        if file_errors:
            files_with_errors += 1
            all_errors.extend(file_errors)

    return {
        "total_label_files": total_label_files,
        "files_with_errors": files_with_errors,
        "total_annotation_lines": total_annotation_lines,
        "empty_label_files": empty_label_files,
        "errors": all_errors[:max_errors_shown],
    }


def check_class_distribution(processed_root: Path) -> pd.DataFrame:
    """
    Compute the per-class annotation frequency across all label files.

    Parameters
    ----------
    processed_root : Path
        Root of the processed data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``class_id`` (int): class index in [0, 14].
        - ``class_name`` (str): human-readable name from ``CLASS_NAMES``.
        - ``count`` (int): number of annotations for this class.
        - ``pct`` (float): percentage of all annotations.

        Rows are ordered by ``class_id``.

    Notes
    -----
    A ``WARNING`` is logged for any class whose count is below 50.
    """
    labels_dir = processed_root / "labels"
    counts: dict[int, int] = {i: 0 for i in range(15)}

    for txt_path in labels_dir.rglob("*.txt"):
        text = txt_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if not tokens:
                continue
            try:
                class_id = int(tokens[0])
            except ValueError:
                continue
            if 0 <= class_id <= 14:
                counts[class_id] += 1

    total = sum(counts.values())

    rows = []
    for class_id in range(15):
        count = counts[class_id]
        pct = (count / total * 100.0) if total > 0 else 0.0
        if count < 50:
            logger.warning(
                "Class %d (%s) has only %d annotations (< 50).",
                class_id,
                CLASS_NAMES[class_id],
                count,
            )
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "count": count,
                "pct": round(pct, 4),
            }
        )

    return pd.DataFrame(rows, columns=["class_id", "class_name", "count", "pct"])


def check_splits(splits_dir: Path, processed_root: Path) -> list[str]:
    """
    Validate the cross-validation fold CSV files.

    Parameters
    ----------
    splits_dir : Path
        Directory containing ``fold_0.csv`` through ``fold_4.csv``.
    processed_root : Path
        Root of the processed data directory (currently unused but kept for
        interface consistency and future cross-referencing).

    Returns
    -------
    list[str]
        Error strings describing any validation failures.  An empty list
        means all fold CSVs are valid.

    Notes
    -----
    Checks performed:

    1. All 5 fold CSVs (``fold_0.csv`` … ``fold_4.csv``) exist.
    2. Each CSV is non-empty.
    3. Each CSV contains a ``split`` column.
    4. The ``split`` column contains both ``"train"`` and ``"val"`` values.
    5. No ``study_id`` appears in both the train and val sets within the
       same fold CSV.
    """
    errors: list[str] = []
    expected_folds = [f"fold_{i}.csv" for i in range(5)]
    missing_folds = [
        f for f in expected_folds if not (splits_dir / f).exists()
    ]
    if missing_folds:
        errors.append(f"Missing fold CSV files: {missing_folds}")

    for fold_name in expected_folds:
        fold_path = splits_dir / fold_name
        if not fold_path.exists():
            # Already reported above
            continue

        try:
            df = pd.read_csv(fold_path)
        except Exception as exc:
            errors.append(f"{fold_path}: failed to read CSV: {exc}")
            continue

        if df.empty:
            errors.append(f"{fold_path}: CSV is empty.")
            continue

        if "split" not in df.columns:
            errors.append(f"{fold_path}: missing 'split' column.")
            continue

        split_values = set(df["split"].unique())
        if "train" not in split_values:
            errors.append(f"{fold_path}: no 'train' rows in 'split' column.")
        if "val" not in split_values:
            errors.append(f"{fold_path}: no 'val' rows in 'split' column.")

        if "study_id" in df.columns:
            train_ids = set(df.loc[df["split"] == "train", "study_id"])
            val_ids = set(df.loc[df["split"] == "val", "study_id"])
            overlap = train_ids & val_ids
            if overlap:
                sample = sorted(overlap)[:5]
                errors.append(
                    f"{fold_path}: {len(overlap)} study_id(s) appear in both "
                    f"train and val sets. Sample: {sample}"
                )

    return errors


def check_image_integrity(
    processed_root: Path, sample_n: int = 100
) -> list[str]:
    """
    Spot-check a random sample of processed PNG images for readability and
    basic content validity.

    Parameters
    ----------
    processed_root : Path
        Root of the processed data directory.
    sample_n : int, optional
        Maximum number of images to sample.  Defaults to 100.

    Returns
    -------
    list[str]
        Error strings for each sampled image that fails a check.  An empty
        list means all sampled images are valid.

    Notes
    -----
    Each sampled image is checked for:

    1. Readability via ``cv2.imread(..., cv2.IMREAD_GRAYSCALE)`` (non-None).
    2. Shape equal to ``(640, 640)``.
    3. Maximum pixel value > 10 (not an all-black image).

    ``random.seed(42)`` is applied before sampling for reproducibility.
    """
    images_dir = processed_root / "images"
    all_pngs = list(images_dir.rglob("*.png"))

    random.seed(42)
    sample = random.sample(all_pngs, min(sample_n, len(all_pngs)))

    errors: list[str] = []
    for png_path in sample:
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors.append(f"{png_path}: cv2.imread returned None (unreadable).")
            continue
        if img.shape != (640, 640):
            errors.append(
                f"{png_path}: shape {img.shape} != (640, 640)."
            )
        if img.max() <= 10:
            errors.append(
                f"{png_path}: max pixel value {img.max()} <= 10 (all-black)."
            )
    return errors


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_checks(processed_root: Path, splits_dir: Path) -> bool:
    """
    Execute all dataset verification checks and print a rich summary table.

    Parameters
    ----------
    processed_root : Path
        Root of the processed data directory.
    splits_dir : Path
        Directory containing the fold CSV split files.

    Returns
    -------
    bool
        ``True`` if all *critical* checks pass, ``False`` otherwise.

    Notes
    -----
    Critical checks (failures cause the function to return ``False``):

    - ``check_image_label_pairing``
    - ``check_all_labels``
    - ``check_splits``

    Non-critical checks (failures emit warnings only):

    - ``check_class_distribution``
    - ``check_image_integrity``
    """
    console = Console()

    # ------------------------------------------------------------------ #
    # Run checks
    # ------------------------------------------------------------------ #
    pairing_errors = check_image_label_pairing(processed_root)

    label_report = check_all_labels(processed_root)
    label_errors = label_report["errors"]
    label_critical_fail = label_report["files_with_errors"] > 0

    splits_errors = check_splits(splits_dir, processed_root)

    dist_df = check_class_distribution(processed_root)

    integrity_errors = check_image_integrity(processed_root)

    # ------------------------------------------------------------------ #
    # Determine pass/fail
    # ------------------------------------------------------------------ #
    critical_pass = (
        len(pairing_errors) == 0
        and not label_critical_fail
        and len(splits_errors) == 0
    )

    # ------------------------------------------------------------------ #
    # Build rich table
    # ------------------------------------------------------------------ #
    table = Table(title="Dataset Verification Report", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Issues", justify="right")
    table.add_column("Summary")

    def _status(ok: bool) -> str:
        return "[green]✓ PASS[/green]" if ok else "[red]✗ FAIL[/red]"

    # Row 1: image/label pairing
    pairing_ok = len(pairing_errors) == 0
    table.add_row(
        "Image–Label Pairing",
        _status(pairing_ok),
        str(len(pairing_errors)),
        "All PNGs have matching .txt files"
        if pairing_ok
        else f"First error: {pairing_errors[0]}",
    )

    # Row 2: label format
    label_format_ok = not label_critical_fail
    table.add_row(
        "Label Format",
        _status(label_format_ok),
        str(label_report["files_with_errors"]),
        (
            f"files={label_report['total_label_files']}, "
            f"annotations={label_report['total_annotation_lines']}, "
            f"empty={label_report['empty_label_files']}"
        ),
    )

    # Row 3: splits
    splits_ok = len(splits_errors) == 0
    table.add_row(
        "Fold Splits",
        _status(splits_ok),
        str(len(splits_errors)),
        "All 5 folds valid"
        if splits_ok
        else f"First error: {splits_errors[0]}",
    )

    # Row 4: class distribution (non-critical)
    low_classes = dist_df[dist_df["count"] < 50]
    dist_ok = len(low_classes) == 0
    dist_summary = (
        f"total classes=15, min count={dist_df['count'].min()}, "
        f"max count={dist_df['count'].max()}"
    )
    table.add_row(
        "Class Distribution",
        _status(dist_ok),
        str(len(low_classes)),
        dist_summary,
    )

    # Row 5: image integrity (non-critical)
    integrity_ok = len(integrity_errors) == 0
    table.add_row(
        "Image Integrity",
        _status(integrity_ok),
        str(len(integrity_errors)),
        "Sample passed all checks"
        if integrity_ok
        else f"First error: {integrity_errors[0]}",
    )

    console.print(table)

    # ------------------------------------------------------------------ #
    # Print detailed errors for failing checks
    # ------------------------------------------------------------------ #
    if pairing_errors:
        console.print("\n[red bold]Image–Label Pairing Errors:[/red bold]")
        for err in pairing_errors[:20]:
            console.print(f"  {err}")

    if label_errors:
        console.print("\n[red bold]Label Format Errors:[/red bold]")
        for err in label_errors:
            console.print(f"  {err}")

    if splits_errors:
        console.print("\n[red bold]Fold Split Errors:[/red bold]")
        for err in splits_errors:
            console.print(f"  {err}")

    if not integrity_ok:
        console.print("\n[yellow bold]Image Integrity Warnings:[/yellow bold]")
        for err in integrity_errors[:20]:
            console.print(f"  {err}")

    if not dist_ok:
        console.print("\n[yellow bold]Class Distribution Warnings:[/yellow bold]")
        for _, row in low_classes.iterrows():
            console.print(
                f"  class {row['class_id']} ({row['class_name']}): "
                f"{row['count']} annotations"
            )

    # ------------------------------------------------------------------ #
    # Final verdict
    # ------------------------------------------------------------------ #
    if critical_pass:
        console.print("\n[green bold]All critical checks PASSED.[/green bold]")
    else:
        console.print(
            "\n[red bold]One or more critical checks FAILED. "
            "Exiting with code 1.[/red bold]"
        )

    return critical_pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Parse CLI arguments and run all dataset verification checks.

    Exits with code 0 on success, code 1 if any critical check fails.
    """
    parser = argparse.ArgumentParser(
        description="Verify the processed RSNA 2024 lumbar spine dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed",
        help="Root directory of the processed images and labels.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing fold_0.csv … fold_4.csv.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=100,
        help="Number of images to sample for integrity check.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    processed_root = Path(args.processed_root)
    splits_dir = Path(args.splits_dir)

    ok = run_all_checks(processed_root, splits_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
