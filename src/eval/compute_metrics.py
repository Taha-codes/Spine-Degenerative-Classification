"""
RSNA 2024 Lumbar Spine Degenerative Classification — evaluation metrics.

Provides helpers to:
- Load and normalise ground-truth labels from train.csv
- Run YOLO validation to extract per-class AP metrics
- Run YOLO inference on a directory of PNGs to collect raw detections
- Convert raw detections to the study-level prediction DataFrame
- Compute the full suite of competition metrics and save results to JSON
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.utils.rsna_metric import rsna_score, predictions_from_yolo, LABEL_COLUMNS
from src.utils.bbox_utils import CLASS_MAP, CLASS_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity string → integer index
# ---------------------------------------------------------------------------
_SEVERITY_MAP: dict[str, int] = {
    "normal/mild": 0,
    "moderate": 1,
    "severe": 2,
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_ground_truth(train_csv_path: Path) -> pd.DataFrame:
    """
    Load and normalise ground-truth labels from train.csv.

    Parameters
    ----------
    train_csv_path : Path
        Path to the RSNA train.csv file (wide format with study_id + 25
        severity columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``study_id`` (str) and the 25 ``LABEL_COLUMNS``
        (int: 0=Normal/Mild, 1=Moderate, 2=Severe, -1=missing).

    Notes
    -----
    Column names in train.csv are normalised before matching:
    lowercased, spaces replaced with underscores, "/" replaced with "_".
    """
    df = pd.read_csv(train_csv_path)

    # Normalise column names to match LABEL_COLUMNS format
    col_rename: dict[str, str] = {}
    for col in df.columns:
        normalised = col.strip().lower().replace(" ", "_").replace("/", "_")
        if normalised != col:
            col_rename[col] = normalised
    if col_rename:
        df = df.rename(columns=col_rename)

    # Ensure study_id is string
    df["study_id"] = df["study_id"].astype(str)

    # Map severity strings → int for each of the 25 label columns
    for label_col in LABEL_COLUMNS:
        if label_col not in df.columns:
            df[label_col] = -1
            continue
        df[label_col] = (
            df[label_col]
            .apply(
                lambda v: _SEVERITY_MAP.get(str(v).strip().lower(), -1)
                if pd.notna(v)
                else -1
            )
            .astype(int)
        )

    return df[["study_id"] + LABEL_COLUMNS]


def run_yolo_inference(
    weights_path: Path,
    dataset_yaml: Path,
    device: str,
    split: str = "val",
) -> pd.DataFrame:
    """
    Run YOLO validation and return per-class AP metrics.

    Parameters
    ----------
    weights_path : Path
        Path to the trained YOLO model weights (.pt file).
    dataset_yaml : Path
        Path to the YOLO dataset YAML file.
    device : str
        Device string (e.g. "cuda", "mps", "cpu").
    split : str, optional
        Dataset split to evaluate on. Default is "val".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``class_id`` (int 0–14), ``class_name`` (str),
        ``ap50`` (float), ``ap50_95`` (float).
        Values are -1.0 if metrics are unavailable.
    """
    model = YOLO(str(weights_path))
    val_results = model.val(data=str(dataset_yaml), split=split, device=device)

    try:
        ap50_list = list(val_results.box.ap50)
    except Exception:
        logger.warning("Could not extract ap50 from val results; defaulting to -1.0")
        ap50_list = [-1.0] * 15

    try:
        ap50_95_list = list(val_results.box.ap)
    except Exception:
        logger.warning("Could not extract ap50_95 from val results; defaulting to -1.0")
        ap50_95_list = [-1.0] * 15

    # Pad or truncate to exactly 15 entries
    ap50_list = (ap50_list + [-1.0] * 15)[:15]
    ap50_95_list = (ap50_95_list + [-1.0] * 15)[:15]

    return pd.DataFrame(
        {
            "class_id": list(range(15)),
            "class_name": CLASS_NAMES,
            "ap50": [float(v) for v in ap50_list],
            "ap50_95": [float(v) for v in ap50_95_list],
        }
    )


def run_yolo_predict(
    weights_path: Path,
    image_dir: Path,
    device: str,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    Run YOLO prediction on all PNG images in a directory.

    Parameters
    ----------
    weights_path : Path
        Path to the trained YOLO model weights (.pt file).
    image_dir : Path
        Directory (searched recursively) containing PNG images.
        Image filenames must follow the pattern:
        ``{study_id}_{series_id}_{instance_number}.png``.
    device : str
        Device string (e.g. "cuda", "mps", "cpu").
    conf_threshold : float, optional
        Minimum confidence threshold for detections. Default is 0.25.

    Returns
    -------
    list[dict]
        Flat list of detection dicts, one dict per detected box across all
        images. Each dict contains: ``study_id``, ``series_id``,
        ``instance_number``, ``class_id``, ``confidence``, ``x_center``,
        ``y_center``, ``width``, ``height`` (last four normalised).
        Images with zero detections contribute no dicts to the list.
    """
    png_list = sorted(image_dir.rglob("*.png"))
    if not png_list:
        logger.warning("No PNG files found in %s", image_dir)
        return []

    model = YOLO(str(weights_path))
    results = model.predict(
        source=[str(p) for p in png_list],
        device=device,
        conf=conf_threshold,
        verbose=False,
    )

    all_detections: list[dict] = []
    for result in results:
        try:
            parts = Path(result.path).stem.split("_")
            study_id = parts[0]
            series_id = parts[1]
            instance_number = parts[2]
        except (IndexError, AttributeError) as exc:
            logger.warning("Could not parse filename %s: %s", result.path, exc)
            continue

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            try:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x_center = float(box.xywhn[0][0].item())
                y_center = float(box.xywhn[0][1].item())
                width = float(box.xywhn[0][2].item())
                height = float(box.xywhn[0][3].item())
            except Exception as exc:
                logger.warning("Could not parse detection box: %s", exc)
                continue

            all_detections.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "instance_number": instance_number,
                    "class_id": class_id,
                    "confidence": confidence,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )

    return all_detections


def detections_to_study_predictions(detections: list[dict]) -> pd.DataFrame:
    """
    Convert a flat list of YOLO detection dicts to a study-level prediction
    DataFrame compatible with ``rsna_score``.

    Parameters
    ----------
    detections : list[dict]
        Flat list of detection dicts as returned by ``run_yolo_predict``.
        Each dict must contain at minimum ``study_id``, ``class_id``, and
        ``confidence``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``study_id`` + 75 probability columns as
        expected by ``predictions_from_yolo`` / ``rsna_score``.

    Notes
    -----
    Limitation: ``run_yolo_predict`` detections do not encode the vertebral
    level (e.g. L1-L2, L2-L3) because the YOLO model predicts condition and
    severity class only, not the spinal level.  As a result, this function
    passes ``level=""`` for every detection.  ``predictions_from_yolo`` will
    find no level match and fall back to uniform probabilities
    ``[1/3, 1/3, 1/3]`` for all 25 label columns.  To obtain meaningful RSNA
    scores, level information must be encoded during dataset preparation and
    post-processing.
    """
    # Collect unique study IDs, preserving insertion order
    study_id_set: dict[str, None] = {}
    for det in detections:
        study_id_set[det["study_id"]] = None
    study_ids = list(study_id_set.keys())

    # Group detections by study_id
    study_det_map: dict[str, list[dict]] = {sid: [] for sid in study_ids}
    for det in detections:
        sid = det["study_id"]
        study_det_map[sid].append(
            {
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "level": "",  # level not available from raw YOLO predictions
            }
        )

    study_detections_list = [study_det_map[sid] for sid in study_ids]
    return predictions_from_yolo(study_detections_list, study_ids)


def compute_full_metrics(
    weights_path: Path,
    dataset_yaml: Path,
    train_csv_path: Path,
    image_dir: Path,
    device: str,
) -> dict[str, Any]:
    """
    Compute the full suite of evaluation metrics for a trained YOLO model.

    Parameters
    ----------
    weights_path : Path
        Path to the trained YOLO model weights (.pt file).
    dataset_yaml : Path
        Path to the YOLO dataset YAML file used during training.
    train_csv_path : Path
        Path to the RSNA train.csv ground-truth file.
    image_dir : Path
        Directory containing validation PNG images (searched recursively).
    device : str
        Device string (e.g. "cuda", "mps", "cpu").

    Returns
    -------
    dict
        Dictionary containing:

        - ``"map50_overall"`` (float): mean AP@0.50 across all 15 classes.
        - ``"map50_95_overall"`` (float): mean AP@[0.50:0.95] across all 15 classes.
        - ``"rsna_score_overall"`` (float): competition score (lower is better).
        - ``"rsna_per_condition"`` (dict[str, float]): score per condition.
        - ``"per_class_ap50"`` (dict[str, float]): AP@0.50 per class name.
        - ``"n_val_images"`` (int): total validation PNG images found.
        - ``"n_val_studies"`` (int): number of unique studies with predictions.
        - ``"weights_path"`` (str): path to model weights used.
    """
    logger.info("Running YOLO validation to extract AP metrics...")
    per_class_df = run_yolo_inference(weights_path, dataset_yaml, device)
    map50_overall = per_class_df["ap50"].mean()
    map50_95_overall = per_class_df["ap50_95"].mean()

    logger.info("Running YOLO prediction on validation images...")
    detections = run_yolo_predict(weights_path, image_dir, device)

    logger.info("Converting detections to study-level predictions...")
    pred_df = detections_to_study_predictions(detections)

    logger.info("Loading ground truth from %s", train_csv_path)
    gt_df = load_ground_truth(train_csv_path)

    logger.info("Computing RSNA competition score...")
    score_dict = rsna_score(gt_df, pred_df)

    n_val_images = len(list(image_dir.rglob("*.png")))
    n_val_studies = int(pred_df["study_id"].nunique())

    return {
        "map50_overall": float(map50_overall),
        "map50_95_overall": float(map50_95_overall),
        "rsna_score_overall": score_dict["overall"],
        "rsna_per_condition": score_dict["per_condition"],
        "per_class_ap50": dict(zip(per_class_df["class_name"], per_class_df["ap50"])),
        "n_val_images": n_val_images,
        "n_val_studies": n_val_studies,
        "weights_path": str(weights_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    CLI entry point for computing evaluation metrics.

    Parse command-line arguments, run ``compute_full_metrics``, save results
    to a JSON file, and print a rich summary table.
    """
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for a trained YOLO spine model."
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to YOLO weights (.pt file).")
    parser.add_argument("--dataset-yaml", type=str, default="configs/dataset.yaml",
                        help="Path to YOLO dataset YAML. Default: configs/dataset.yaml")
    parser.add_argument("--train-csv", type=str, required=True,
                        help="Path to RSNA train.csv ground-truth file.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing validation PNG images.")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to run inference on. Default: cuda")
    parser.add_argument("--output-json", type=str, default="runs/eval_results.json",
                        help="Path for output JSON file. Default: runs/eval_results.json")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO predictions. Default: 0.25")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = compute_full_metrics(
        weights_path=Path(args.weights),
        dataset_yaml=Path(args.dataset_yaml),
        train_csv_path=Path(args.train_csv),
        image_dir=Path(args.image_dir),
        device=args.device,
    )

    # Save to JSON
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to %s", output_json)

    # Rich summary table
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console()
        table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold", min_width=35)
        table.add_column("Value", justify="right", min_width=15)

        # Key scalar metrics
        table.add_row("mAP@50 (overall)", f"{results['map50_overall']:.4f}")
        table.add_row("mAP@50-95 (overall)", f"{results['map50_95_overall']:.4f}")
        table.add_row("Validation images", str(results["n_val_images"]))
        table.add_row("Validation studies", str(results["n_val_studies"]))

        # Per-class AP50
        table.add_section()
        for cls_name, ap50_val in results["per_class_ap50"].items():
            table.add_row(f"  AP50  {cls_name}", f"{ap50_val:.4f}")

        # Per-condition RSNA scores
        table.add_section()
        for cond, cond_score in results["rsna_per_condition"].items():
            table.add_row(f"  RSNA  {cond}", f"{cond_score:.4f}")

        # Highlight the overall competition metric
        table.add_section()
        rsna_overall = results["rsna_score_overall"]
        table.add_row(
            Text("RSNA SCORE (OVERALL)", style="bold yellow"),
            Text(f"{rsna_overall:.4f}", style="bold yellow"),
        )

        console.print(table)
        console.print(
            f"\n[bold green]Competition metric (RSNA score overall):[/bold green] "
            f"[bold yellow]{rsna_overall:.6f}[/bold yellow]  "
            f"[dim](lower is better)[/dim]"
        )
        console.print(f"[dim]Results saved to: {output_json}[/dim]")

    except ImportError:
        # Fallback if rich is not available
        print("\n=== Evaluation Results ===")
        print(f"  mAP@50 (overall)       : {results['map50_overall']:.4f}")
        print(f"  mAP@50-95 (overall)    : {results['map50_95_overall']:.4f}")
        print(f"  RSNA score (OVERALL)   : {results['rsna_score_overall']:.6f}  [competition metric, lower=better]")
        print(f"  Validation images      : {results['n_val_images']}")
        print(f"  Validation studies     : {results['n_val_studies']}")
        print("\n  Per-condition RSNA scores:")
        for cond, cond_score in results["rsna_per_condition"].items():
            print(f"    {cond}: {cond_score:.4f}")
        print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
