"""
Stage 4 – DICOM to PNG preprocessing for the RSNA 2024 Lumbar Spine pipeline.

Converts raw DICOM files to 640×640 uint8 PNG images, routing each series to
the appropriate output subdirectory based on the series description.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.dicom_utils import dicom_to_numpy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SERIES_ROUTING: dict[str, str] = {
    "t1":    "sagittal_t1",
    "t2":    "sagittal_t2",
    "stir":  "sagittal_t2",
    "axial": "axial_t2",
    "ax":    "axial_t2",
}

# Routing priority order: more-specific (longer) keys are checked first so
# that "axial t2" routes to axial_t2 rather than sagittal_t2.
_ROUTING_PRIORITY: list[tuple[str, str]] = sorted(
    SERIES_ROUTING.items(), key=lambda kv: len(kv[0]), reverse=True
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def route_series(series_description: str) -> str:
    """
    Map a DICOM series description string to a canonical series type.

    Lowercases the input, then checks whether any key in ``SERIES_ROUTING``
    appears as a substring.  The first matching key wins.

    Parameters
    ----------
    series_description : str
        Raw series description string (e.g. ``"Sagittal T2/STIR"``).

    Returns
    -------
    str
        One of ``"sagittal_t1"``, ``"sagittal_t2"``, ``"axial_t2"``, or
        ``"unknown"`` when no key matches.
    """
    lower = series_description.lower()
    for key, canonical in _ROUTING_PRIORITY:
        if key in lower:
            return canonical
    logger.warning(
        "route_series: no routing match for series description %r; "
        "returning 'unknown'",
        series_description,
    )
    return "unknown"


def pad_and_resize(arr: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Scale and center-pad a 2-D uint8 image to a square canvas.

    The longest side is scaled to ``target_size`` (preserving aspect ratio),
    then the result is placed on a black square canvas of
    ``target_size × target_size``.

    Parameters
    ----------
    arr : np.ndarray
        2-D uint8 input image.
    target_size : int, optional
        Side length of the output square canvas in pixels.  Default is 640.

    Returns
    -------
    np.ndarray
        uint8 2-D array of shape ``(target_size, target_size)``.
    """
    h, w = arr.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Per-study processing
# ---------------------------------------------------------------------------

def process_study(
    study_id: str,
    raw_root: str | Path,
    output_root: str | Path,
    target_size: int = 640,
    apply_clahe_flag: bool = True,
) -> List[dict]:
    """
    Convert all DICOM files for a single study to PNG.

    Discovers every ``*.dcm`` file under
    ``raw_root/train_images/{study_id}/``, converts each to a padded
    grayscale PNG, and writes it to
    ``output_root/{series_type}/{study_id}_{series_id}_{instance_number}.png``.

    Parameters
    ----------
    study_id : str
        Study identifier (directory name under ``train_images/``).
    raw_root : str or pathlib.Path
        Root directory that contains the ``train_images/`` folder.
    output_root : str or pathlib.Path
        Root directory for PNG output.
    target_size : int, optional
        Square canvas size in pixels.  Default is 640.
    apply_clahe_flag : bool, optional
        Whether to apply CLAHE during DICOM loading.  Default is True.

    Returns
    -------
    list of dict
        Each dict has keys: ``study_id``, ``series_id``, ``instance_number``,
        ``series_type``, ``output_path``, ``original_rows``,
        ``original_cols``, ``pixel_spacing``.
    """
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    study_dir = raw_root / "train_images" / str(study_id)
    dcm_files = list(study_dir.rglob("*.dcm"))

    records: List[dict] = []

    for path in dcm_files:
        try:
            arr, meta = dicom_to_numpy(
                path,
                apply_window=True,
                apply_clahe_flag=apply_clahe_flag,
            )
            series_type = route_series(meta["series_description"])
            padded_arr = pad_and_resize(arr, target_size)

            series_id = path.parent.name
            instance_number = path.stem

            output_dir = output_root / series_type
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{study_id}_{series_id}_{instance_number}.png"
            cv2.imwrite(str(output_path), padded_arr)

            records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "instance_number": instance_number,
                    "series_type": series_type,
                    "output_path": str(output_path),
                    "original_rows": meta["rows"],
                    "original_cols": meta["cols"],
                    "pixel_spacing": meta["pixel_spacing"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("process_study: skipping %s – %s", path, exc)

    logger.info("study %s: processed %d files", study_id, len(records))
    return records


# ---------------------------------------------------------------------------
# Module-level wrapper required by ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------

def _process_study_worker(
    study_id: str,
    raw_root: str,
    output_root: str,
    target_size: int,
    apply_clahe_flag: bool,
) -> List[dict]:
    """Top-level wrapper so ProcessPoolExecutor can pickle the callable."""
    return process_study(
        study_id=study_id,
        raw_root=raw_root,
        output_root=output_root,
        target_size=target_size,
        apply_clahe_flag=apply_clahe_flag,
    )


# ---------------------------------------------------------------------------
# All-studies orchestration
# ---------------------------------------------------------------------------

def process_all_studies(
    raw_root: str | Path,
    output_root: str | Path,
    target_size: int = 640,
    n_workers: int = 4,
    apply_clahe_flag: bool = True,
) -> pd.DataFrame:
    """
    Convert every study in the dataset from DICOM to PNG in parallel.

    Discovers all immediate subdirectories of ``raw_root/train_images/``
    and processes each with a ``ProcessPoolExecutor``.  A tqdm progress
    bar tracks completion.

    Parameters
    ----------
    raw_root : str or pathlib.Path
        Root directory that contains the ``train_images/`` folder.
    output_root : str or pathlib.Path
        Root directory for PNG output.
    target_size : int, optional
        Square canvas size in pixels.  Default is 640.
    n_workers : int, optional
        Number of worker processes.  Default is 4.
    apply_clahe_flag : bool, optional
        Whether to apply CLAHE during DICOM loading.  Default is True.

    Returns
    -------
    pd.DataFrame
        Combined metadata for every successfully converted image with
        columns: ``study_id``, ``series_id``, ``instance_number``,
        ``series_type``, ``output_path``, ``original_rows``,
        ``original_cols``, ``pixel_spacing``.
    """
    raw_root = Path(raw_root)
    train_images_dir = raw_root / "train_images"
    study_ids = [d.name for d in train_images_dir.iterdir() if d.is_dir()]

    worker = functools.partial(
        _process_study_worker,
        raw_root=str(raw_root),
        output_root=str(output_root),
        target_size=target_size,
        apply_clahe_flag=apply_clahe_flag,
    )

    all_records: List[dict] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_study = {executor.submit(worker, sid): sid for sid in study_ids}

        with tqdm(
            concurrent.futures.as_completed(future_to_study),
            total=len(future_to_study),
            desc="Studies",
            unit="study",
        ) as pbar:
            for future in pbar:
                sid = future_to_study[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "process_all_studies: study %s failed – %s", sid, exc
                    )

    if all_records:
        return pd.DataFrame(all_records)
    return pd.DataFrame(
        columns=[
            "study_id",
            "series_id",
            "instance_number",
            "series_type",
            "output_path",
            "original_rows",
            "original_cols",
            "pixel_spacing",
        ]
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface for DICOM-to-PNG conversion.

    Examples
    --------
    Process all studies::

        python -m src.preprocess.dicom_to_png \\
            --raw-root data/raw \\
            --output-root data/processed

    Debug a single study::

        python -m src.preprocess.dicom_to_png \\
            --raw-root data/raw \\
            --study-id 4003253
    """
    parser = argparse.ArgumentParser(
        description="Convert RSNA DICOM files to 640×640 grayscale PNGs."
    )
    parser.add_argument("--raw-root", required=True, type=str,
                        help="Root directory containing train_images/")
    parser.add_argument("--output-root", default="data/processed", type=str,
                        help="Root directory for PNG output (default: data/processed)")
    parser.add_argument("--target-size", default=640, type=int,
                        help="Square canvas side length in pixels (default: 640)")
    parser.add_argument("--workers", default=4, type=int,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Disable CLAHE contrast enhancement")
    parser.add_argument("--study-id", default=None, type=str,
                        help="Process a single study ID (debug mode; no manifest saved)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    apply_clahe = not args.no_clahe

    if args.study_id is not None:
        records = process_study(
            study_id=args.study_id,
            raw_root=args.raw_root,
            output_root=args.output_root,
            target_size=args.target_size,
            apply_clahe_flag=apply_clahe,
        )
        for rec in records:
            print(rec)
    else:
        df = process_all_studies(
            raw_root=args.raw_root,
            output_root=args.output_root,
            target_size=args.target_size,
            n_workers=args.workers,
            apply_clahe_flag=apply_clahe,
        )
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.csv"
        df.to_csv(manifest_path, index=False)
        logger.info("Manifest saved to %s (%d rows)", manifest_path, len(df))


if __name__ == "__main__":
    main()
