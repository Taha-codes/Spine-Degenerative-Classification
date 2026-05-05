"""
Stage 9 — Smoke test: 3-epoch pipeline verification for RSNA 2024 Lumbar Spine.

Runs a short YOLOv11 training pass on both yolo11n and yolo11m to confirm that
the dataset, hyperparameters, and MPS device are wired up correctly before
submitting full training runs.
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_mps_available() -> bool:
    """Check whether Apple MPS is available and built.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True only when both ``torch.backends.mps.is_available()`` and
        ``torch.backends.mps.is_built()`` return True.
    """
    result = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    logger.info("MPS available: %s", result)
    return result


def check_dataset(dataset_yaml: Path) -> bool:
    """Validate a YOLO dataset YAML and its on-disk layout.

    Parameters
    ----------
    dataset_yaml : Path
        Absolute or relative path to the dataset YAML file.

    Returns
    -------
    bool
        True only when ALL of the following conditions hold:

        * The YAML file exists on disk.
        * The YAML contains a ``path`` key whose value is an existing directory.
        * ``images/train`` and ``images/val`` subdirectories exist inside that
          directory and each contain at least one PNG file.
        * The ``nc`` key equals 15.
    """
    # 1. File existence
    if not dataset_yaml.exists():
        logger.error("Dataset YAML not found: %s", dataset_yaml)
        return False

    with dataset_yaml.open() as fh:
        cfg = yaml.safe_load(fh)

    # 2. 'path' key and directory existence
    if "path" not in cfg:
        logger.error("Dataset YAML missing 'path' key: %s", dataset_yaml)
        return False

    dataset_root = Path(cfg["path"])
    if not dataset_root.exists():
        logger.error("Dataset 'path' directory does not exist: %s", dataset_root)
        return False

    # 3. images/train and images/val subdirectories with at least 1 PNG each
    for split in ("train", "val"):
        split_dir = dataset_root / "images" / split
        if not split_dir.exists():
            logger.error("Missing split directory: %s", split_dir)
            return False
        pngs = list(split_dir.glob("*.png"))
        if len(pngs) < 1:
            logger.error("No PNG files found in: %s", split_dir)
            return False

    # 4. nc == 15
    if cfg.get("nc") != 15:
        logger.error(
            "Expected nc=15 in dataset YAML, got nc=%s", cfg.get("nc")
        )
        return False

    logger.info("Dataset check passed: %s", dataset_yaml)
    return True


def run_smoke_test(
    model_name: str,
    dataset_yaml: Path,
    hyp_yaml: Path,
    project_root: Path,
    device: str = "mps",
) -> dict:
    """Run a short smoke-test training pass for one YOLO model.

    Parameters
    ----------
    model_name : str
        Base model name without extension, e.g. ``"yolo11n"`` or
        ``"yolo11m"``.
    dataset_yaml : Path
        Path to the YOLO dataset YAML file.
    hyp_yaml : Path
        Path to the smoke-test hyperparameter YAML file.
    project_root : Path
        Root directory of the project; training artefacts are written to
        ``<project_root>/runs/smoke/<model_name>/``.
    device : str, optional
        Device string passed to ``model.train()``.  Defaults to ``"mps"``.

    Returns
    -------
    dict
        Always returns a dictionary.  On success the dict contains:

        ``model`` : str
            The model name.
        ``success`` : bool
            ``True``.
        ``epochs_completed`` : int
            Number of epochs that finished.
        ``final_box_loss`` : float
            Box regression loss at the last epoch.
        ``final_cls_loss`` : float
            Classification loss at the last epoch.
        ``final_map50`` : float
            mAP@0.50 at the last epoch.
        ``duration_seconds`` : float
            Wall-clock seconds elapsed.
        ``device`` : str
            Device used.

        On failure the dict contains ``"success": False`` and ``"error"``
        with the exception message.
    """
    start = time.time()

    try:
        with hyp_yaml.open() as fh:
            hyp = yaml.safe_load(fh)

        logger.info("Starting smoke test for model: %s on device: %s", model_name, device)
        model = YOLO(f"{model_name}.pt")

        results = model.train(
            device=device,
            data=str(dataset_yaml),
            project=str(project_root / "runs" / "smoke"),
            name=model_name,
            **hyp,
        )

        elapsed = time.time() - start

        # Extract metrics — each field is guarded independently
        try:
            epochs_completed = int(results.epoch + 1)
        except Exception:
            try:
                epochs_completed = int(hyp.get("epochs", 3))
            except Exception:
                epochs_completed = 0

        try:
            final_box_loss = float(results.results_dict.get("train/box_loss", 0.0))
        except Exception:
            final_box_loss = 0.0

        try:
            final_cls_loss = float(results.results_dict.get("train/cls_loss", 0.0))
        except Exception:
            final_cls_loss = 0.0

        try:
            final_map50 = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
        except Exception:
            try:
                final_map50 = float(results.box.map50)
            except Exception:
                final_map50 = 0.0

        logger.info(
            "Smoke test complete — model=%s epochs=%d box_loss=%.4f cls_loss=%.4f mAP50=%.4f duration=%.1fs",
            model_name,
            epochs_completed,
            final_box_loss,
            final_cls_loss,
            final_map50,
            elapsed,
        )

        return {
            "model": model_name,
            "success": True,
            "epochs_completed": epochs_completed,
            "final_box_loss": final_box_loss,
            "final_cls_loss": final_cls_loss,
            "final_map50": final_map50,
            "duration_seconds": elapsed,
            "device": device,
        }

    except Exception as e:
        elapsed = time.time() - start
        logging.exception("Smoke test FAILED for model %s", model_name)
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "duration_seconds": elapsed,
            "device": device,
        }


def print_smoke_report(results: list[dict]) -> None:
    """Print a formatted Rich table summarising smoke-test results.

    Parameters
    ----------
    results : list[dict]
        List of result dicts as returned by :func:`run_smoke_test`.

    Returns
    -------
    None
    """
    console = Console()

    table = Table(title="Smoke Test Results", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Epochs", justify="right")
    table.add_column("Box Loss", justify="right")
    table.add_column("Cls Loss", justify="right")
    table.add_column("mAP50", justify="right")
    table.add_column("Duration(s)", justify="right")
    table.add_column("Device", justify="center")

    n_failed = 0
    for r in results:
        if r.get("success"):
            status = "[green]✓ PASS[/green]"
            epochs = str(r.get("epochs_completed", "-"))
            box_loss = f"{r.get('final_box_loss', 0.0):.4f}"
            cls_loss = f"{r.get('final_cls_loss', 0.0):.4f}"
            map50 = f"{r.get('final_map50', 0.0):.4f}"
        else:
            status = "[red]✗ FAIL[/red]"
            error_msg = r.get("error", "unknown error")
            epochs = f"[red]{error_msg}[/red]"
            box_loss = "-"
            cls_loss = "-"
            map50 = "-"
            n_failed += 1

        table.add_row(
            r.get("model", "-"),
            status,
            epochs,
            box_loss,
            cls_loss,
            map50,
            f"{r.get('duration_seconds', 0.0):.1f}",
            r.get("device", "-"),
        )

    console.print(table)

    if n_failed == 0:
        console.print("[green]✓ All smoke tests passed[/green]")
    else:
        console.print(
            f"[red]✗ {n_failed} smoke test(s) failed — fix before running on Kaggle[/red]"
        )


def main() -> None:
    """CLI entry point for the Stage 9 smoke test.

    Parses arguments, validates MPS availability and the dataset, then runs
    a short training pass for each requested model and prints a summary table.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Stage 9 — 3-epoch smoke test for RSNA Lumbar Spine pipeline"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n", "yolo11m"],
        help="List of model base names to test (default: yolo11n yolo11m)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset YAML (default: configs/dataset.yaml)",
    )
    parser.add_argument(
        "--hyp-yaml",
        type=str,
        default="configs/hyp_smoke.yaml",
        help="Path to smoke hyperparameter YAML (default: configs/hyp_smoke.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: .)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model override — if given, replaces --models",
    )

    args = parser.parse_args()

    overall_start = time.time()
    logger.info("=== Stage 9 Smoke Test — start ===")

    # Step 2: MPS check
    mps_ok = check_mps_available()
    if not mps_ok:
        logger.warning(
            "MPS not available — falling back to CPU. Smoke test will be slow."
        )
    device = "mps" if mps_ok else "cpu"

    # Step 3: Dataset check
    dataset_yaml = Path(args.dataset_yaml)
    dataset_ok = check_dataset(dataset_yaml)
    if not dataset_ok:
        logger.error("Dataset check failed — fix dataset before smoke test")
        sys.exit(1)

    # Step 4: Resolve model list
    models = [args.model] if args.model else args.models

    hyp_yaml = Path(args.hyp_yaml)
    project_root = Path(args.project_root)

    # Step 5: Sequential smoke tests (MPS can't handle concurrent training)
    results = []
    for model_name in models:
        result = run_smoke_test(
            model_name=model_name,
            dataset_yaml=dataset_yaml,
            hyp_yaml=hyp_yaml,
            project_root=project_root,
            device=device,
        )
        results.append(result)

    # Step 6: Print report
    print_smoke_report(results)

    # Step 7: Log total duration
    total_elapsed = time.time() - overall_start
    logger.info("=== Stage 9 Smoke Test — total duration: %.1fs ===", total_elapsed)

    # Step 8: Exit code
    all_passed = all(r.get("success", False) for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
