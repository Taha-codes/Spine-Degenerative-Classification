"""
Stage 10 — Single-fold YOLO training script.

Builds a fold-specific dataset YAML (with symlinked images/labels),
trains a YOLO model, optionally validates, and saves results to JSON.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

# Reference only — not used for logic
from src.utils.rsna_metric import LABEL_COLUMNS  # noqa: F401

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

console = Console()


# ---------------------------------------------------------------------------
# build_fold_dataset_yaml
# ---------------------------------------------------------------------------

def build_fold_dataset_yaml(
    fold_k: int,
    splits_dir: Path,
    processed_root: Path,
    base_dataset_yaml: Path,
    output_dir: Path,
) -> Path:
    """Build a fold-specific dataset YAML with symlinked images and labels.

    Parameters
    ----------
    fold_k : int
        Fold index (0-4).
    splits_dir : Path
        Directory containing fold_{k}.csv files.
    processed_root : Path
        Root of the processed dataset (contains images/ and labels/ sub-dirs).
    base_dataset_yaml : Path
        Base dataset YAML with nc and names fields.
    output_dir : Path
        Root directory where fold sub-directories will be created.

    Returns
    -------
    Path
        Path to the written fold dataset YAML file.
    """
    # 1. Load fold CSV
    fold_csv = splits_dir / f"fold_{fold_k}.csv"
    df = pd.read_csv(fold_csv)
    df["study_id"] = df["study_id"].astype(str)

    # 2. Build train/val id sets
    train_ids: set[str] = set(df.loc[df["split"] == "train", "study_id"])
    val_ids: set[str] = set(df.loc[df["split"] == "val", "study_id"])

    fold_root = output_dir / f"fold_{fold_k}"
    n_train = 0
    n_val = 0

    def _link_or_copy(src: Path, dst: Path) -> None:
        """Create a symlink from dst → src, falling back to copy on failure."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src.resolve(), dst)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(src, dst)

    # 3. Process images
    for png in processed_root.glob("images/**/*.png"):
        stem = png.stem
        study_id = stem.split("_")[0]
        series_type = png.parent.name  # e.g. "sagittal_t1"
        filename = png.name

        if study_id in train_ids:
            dst = fold_root / "images" / "train" / series_type / filename
            _link_or_copy(png, dst)
            n_train += 1
        elif study_id in val_ids:
            dst = fold_root / "images" / "val" / series_type / filename
            _link_or_copy(png, dst)
            n_val += 1

    # 4. Process labels
    for txt in processed_root.glob("labels/**/*.txt"):
        stem = txt.stem
        study_id = stem.split("_")[0]
        series_type = txt.parent.name
        filename = txt.name

        if study_id in train_ids:
            dst = fold_root / "labels" / "train" / series_type / filename
            _link_or_copy(txt, dst)
        elif study_id in val_ids:
            dst = fold_root / "labels" / "val" / series_type / filename
            _link_or_copy(txt, dst)

    # 5. Build and write YAML
    with open(base_dataset_yaml, "r") as fh:
        base_cfg: dict = yaml.safe_load(fh)

    fold_cfg = dict(base_cfg)  # shallow copy keeps nc and names
    fold_cfg["path"] = str(fold_root.resolve())
    fold_cfg["train"] = "images/train"
    fold_cfg["val"] = "images/val"

    out_yaml = fold_root / f"fold_{fold_k}_dataset.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as fh:
        yaml.dump(fold_cfg, fh, default_flow_style=False, sort_keys=False)

    logger.info("Fold %d: %d train images, %d val images", fold_k, n_train, n_val)
    return out_yaml


# ---------------------------------------------------------------------------
# load_class_weights
# ---------------------------------------------------------------------------

def load_class_weights(weights_path: Path) -> dict[int, float]:
    """Load per-class weights from a JSON file.

    Parameters
    ----------
    weights_path : Path
        Path to the JSON file whose keys are string class IDs and values
        are float weights.

    Returns
    -------
    dict[int, float]
        Mapping from integer class ID to float weight.
    """
    with open(weights_path, "r") as fh:
        raw: dict[str, float] = json.load(fh)

    weights: dict[int, float] = {int(k): float(v) for k, v in raw.items()}

    values = list(weights.values())
    values_sorted = sorted(values)
    n = len(values_sorted)
    if n == 0:
        median_w = 0.0
    elif n % 2 == 1:
        median_w = values_sorted[n // 2]
    else:
        median_w = (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2.0

    logger.info(
        "Class weights — min=%.4f, max=%.4f, median=%.4f",
        min(values) if values else 0.0,
        max(values) if values else 0.0,
        median_w,
    )
    return weights


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------

def run_training(
    model_name: str,
    fold_k: int,
    dataset_yaml: Path,
    hyp_yaml: Path,
    class_weights: dict[int, float],
    project_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train a YOLO model on one fold.

    Parameters
    ----------
    model_name : str
        YOLO model variant name (e.g. "yolo11n").
    fold_k : int
        Fold index (0-4).
    dataset_yaml : Path
        Path to the fold-specific dataset YAML.
    hyp_yaml : Path
        Path to the hyperparameter YAML.
    class_weights : dict[int, float]
        Per-class weights (loaded for reference; logged as available for
        custom callbacks, not passed to model.train directly).
    project_dir : Path
        Root directory for Ultralytics run output.
    device : str
        Device string, e.g. "cuda", "mps", or "cpu".

    Returns
    -------
    dict[str, Any]
        Training summary with metrics, paths, and timing.
    """
    start = time.time()
    try:
        with open(hyp_yaml, "r") as fh:
            hyp: dict = yaml.safe_load(fh)

        name = f"{model_name}_fold{fold_k}"
        model = YOLO(f"{model_name}.pt")

        logger.info(
            "Starting training: model=%s, fold=%d, device=%s", model_name, fold_k, device
        )
        logger.info(
            "Class weights available for custom callbacks if needed: %s",
            {k: round(v, 4) for k, v in list(class_weights.items())[:5]},
        )

        model.train(
            device=device,
            data=str(dataset_yaml),
            project=str(project_dir),
            name=name,
            exist_ok=True,
            **hyp,
        )

        # Parse results CSV
        results_csv = project_dir / name / "results.csv"
        df = pd.read_csv(results_csv)

        def _safe_max(col: str) -> float:
            try:
                return float(df[col].max())
            except Exception:
                return -1.0

        def _safe_last(col: str) -> float:
            try:
                return float(df[col].iloc[-1])
            except Exception:
                return -1.0

        def _safe_len() -> int:
            try:
                return int(len(df))
            except Exception:
                return -1

        best_map50 = _safe_max("metrics/mAP50(B)")
        best_map50_95 = _safe_max("metrics/mAP50-95(B)")
        final_box_loss = _safe_last("val/box_loss")
        final_cls_loss = _safe_last("val/cls_loss")
        final_dfl_loss = _safe_last("val/dfl_loss")
        epochs_trained = _safe_len()

        best_weights_path = str((project_dir / name / "weights" / "best.pt").resolve())

        return {
            "model": model_name,
            "fold": fold_k,
            "device": device,
            "best_map50": best_map50,
            "best_map50_95": best_map50_95,
            "final_box_loss": final_box_loss,
            "final_cls_loss": final_cls_loss,
            "final_dfl_loss": final_dfl_loss,
            "epochs_trained": epochs_trained,
            "best_weights_path": best_weights_path,
            "duration_seconds": time.time() - start,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    except Exception as e:  # noqa: BLE001
        logger.error("Training failed: %s", e, exc_info=True)
        return {
            "model": model_name,
            "fold": fold_k,
            "device": device,
            "error": str(e),
            "best_map50": -1,
            "best_map50_95": -1,
            "final_box_loss": -1,
            "final_cls_loss": -1,
            "final_dfl_loss": -1,
            "epochs_trained": -1,
            "best_weights_path": "",
            "duration_seconds": time.time() - start,
            "timestamp": datetime.datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# run_validation
# ---------------------------------------------------------------------------

def run_validation(
    model_name: str,
    fold_k: int,
    best_weights_path: Path,
    dataset_yaml: Path,
    device: str,
) -> dict[str, Any]:
    """Run validation using the best checkpoint from training.

    Parameters
    ----------
    model_name : str
        YOLO model variant name used during training.
    fold_k : int
        Fold index (0-4).
    best_weights_path : Path
        Path to the best.pt weights file.
    dataset_yaml : Path
        Path to the fold-specific dataset YAML.
    device : str
        Device string, e.g. "cuda", "mps", or "cpu".

    Returns
    -------
    dict[str, Any]
        Validation metrics including mAP50, mAP50-95, and per-class AP50.
    """
    model = YOLO(str(best_weights_path))
    val_results = model.val(data=str(dataset_yaml), device=device, split="val")

    try:
        map50 = float(val_results.box.map50)
    except Exception:
        map50 = -1.0

    try:
        map50_95 = float(val_results.box.map)
    except Exception:
        map50_95 = -1.0

    try:
        per_class_ap50: dict[int, float] = {
            int(i): float(v) for i, v in enumerate(val_results.box.ap50)
        }
    except Exception:
        per_class_ap50 = {}

    return {
        "model": model_name,
        "fold": fold_k,
        "map50": map50,
        "map50_95": map50_95,
        "per_class_ap50": per_class_ap50,
    }


# ---------------------------------------------------------------------------
# save_run_results
# ---------------------------------------------------------------------------

def save_run_results(results: dict[str, Any], output_path: Path) -> None:
    """Persist run results to a JSON file.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary of training (and optionally validation) metrics.
    output_path : Path
        Destination path for the JSON file.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for single-fold YOLO training.

    Parses arguments, builds the fold dataset YAML, runs training,
    optionally validates, and saves a JSON results file.
    """
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on a single fold of the RSNA 2024 dataset."
    )
    parser.add_argument("--model", required=True, choices=["yolo11n", "yolo11m"])
    parser.add_argument("--fold", required=True, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    parser.add_argument("--hyp-yaml", default="configs/hyp_base.yaml")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--weights-dir", default="data")
    parser.add_argument("--project-dir", default="runs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--output-json", default=None)

    args = parser.parse_args()

    model: str = args.model
    fold_k: int = args.fold

    # Resolve paths relative to cwd
    splits_dir = Path(args.splits_dir)
    processed_root = Path(args.processed_root)
    base_dataset_yaml = Path(args.dataset_yaml)
    hyp_yaml = Path(args.hyp_yaml)
    weights_dir = Path(args.weights_dir)
    project_dir = Path(args.project_dir)

    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else Path(f"runs/{model}_fold{fold_k}_results.json")
    )

    # 1. Build fold dataset yaml
    fold_dataset_yaml = build_fold_dataset_yaml(
        fold_k=fold_k,
        splits_dir=splits_dir,
        processed_root=processed_root,
        base_dataset_yaml=base_dataset_yaml,
        output_dir=project_dir / "datasets",
    )

    # 2. Load class weights
    class_weights = load_class_weights(weights_dir / "class_weights.json")

    # 3. Training
    train_results = run_training(
        model_name=model,
        fold_k=fold_k,
        dataset_yaml=fold_dataset_yaml,
        hyp_yaml=hyp_yaml,
        class_weights=class_weights,
        project_dir=project_dir,
        device=args.device,
    )

    # 4. Validation (unless skipped or training errored)
    combined: dict[str, Any] = dict(train_results)
    training_ok = "error" not in train_results

    if not args.skip_val and training_ok:
        best_weights = Path(train_results["best_weights_path"])
        val_results = run_validation(
            model_name=model,
            fold_k=fold_k,
            best_weights_path=best_weights,
            dataset_yaml=fold_dataset_yaml,
            device=args.device,
        )
        combined.update(val_results)

    # 5. Save
    save_run_results(combined, output_json)

    # 6. Rich summary table
    table = Table(title=f"Training Summary — {model} fold {fold_k}", show_lines=True)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="green")

    summary_fields = [
        ("model", str(combined.get("model", ""))),
        ("fold", str(combined.get("fold", ""))),
        ("device", str(combined.get("device", ""))),
        ("best_map50", f"{combined.get('best_map50', -1):.4f}"),
        ("best_map50_95", f"{combined.get('best_map50_95', -1):.4f}"),
        ("epochs_trained", str(combined.get("epochs_trained", -1))),
        ("duration_seconds", f"{combined.get('duration_seconds', 0):.1f}s"),
        ("output_json", str(output_json)),
    ]
    if "error" in combined:
        summary_fields.append(("ERROR", combined["error"]))

    for field, value in summary_fields:
        table.add_row(field, value)

    console.print(table)


if __name__ == "__main__":
    main()
