"""
Ablation training orchestrator for the RSNA 2024 Lumbar Spine Degenerative
Classification pipeline (Stage 11).

Runs train_single across a matrix of models x folds, collects results, and
prints a summary table with a model recommendation.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODELS: list[str] = ["yolo11n", "yolo11m"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_run_matrix(models: list[str], folds: list[int]) -> list[dict]:
    """Build the ordered list of (model, fold) runs to execute.

    Parameters
    ----------
    models : list[str]
        Model names, e.g. ["yolo11n", "yolo11m"].
    folds : list[int]
        Fold indices to run for every model, e.g. [0, 1, 2, 3, 4].

    Returns
    -------
    list[dict]
        Ordered list of run dicts.  All folds for models[0] appear before all
        folds for models[1], etc.  Each dict has keys:
        ``model``, ``fold``, ``run_id``.

    Examples
    --------
    >>> matrix = build_run_matrix(["yolo11n", "yolo11m"], [0, 1])
    >>> matrix[0]
    {'model': 'yolo11n', 'fold': 0, 'run_id': 'yolo11n_fold0'}
    """
    matrix: list[dict] = []
    for model in models:
        for fold in folds:
            matrix.append(
                {
                    "model": model,
                    "fold": fold,
                    "run_id": f"{model}_fold{fold}",
                }
            )
    return matrix


def run_single_training(run: dict, cli_args: argparse.Namespace) -> dict:
    """Launch a single model+fold training as a subprocess.

    Parameters
    ----------
    run : dict
        A single entry from ``build_run_matrix``, with keys ``model``,
        ``fold``, and ``run_id``.
    cli_args : argparse.Namespace
        Parsed CLI arguments from ``main()``.

    Returns
    -------
    dict
        The JSON result dict written by ``train_single``, or an error dict
        with ``success=False`` if the subprocess failed or produced no output
        file.

    Notes
    -----
    - ``stdout`` is *not* captured so training progress streams live.
    - ``stderr`` is captured and logged on non-zero exit codes.
    - If ``--resume`` is set and the output JSON already exists the run is
      skipped immediately.
    """
    output_json = Path(cli_args.project_dir) / f"{run['run_id']}_results.json"

    # ------------------------------------------------------------------
    # Resume / skip logic
    # ------------------------------------------------------------------
    if getattr(cli_args, "resume", False) and output_json.exists():
        logger.info("Skipping %s — already completed", run["run_id"])
        with output_json.open() as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Build CLI command
    # ------------------------------------------------------------------
    cmd = [
        sys.executable,
        "-m",
        "src.train.train_single",
        "--model",
        run["model"],
        "--fold",
        str(run["fold"]),
        "--device",
        cli_args.device,
        "--dataset-yaml",
        cli_args.dataset_yaml,
        "--hyp-yaml",
        cli_args.hyp_yaml,
        "--splits-dir",
        cli_args.splits_dir,
        "--processed-root",
        cli_args.processed_root,
        "--project-dir",
        cli_args.project_dir,
        "--output-json",
        str(output_json),
    ]

    logger.info("Launching: %s", " ".join(cmd))

    # ------------------------------------------------------------------
    # Execute — stdout streams live, stderr captured
    # ------------------------------------------------------------------
    result = subprocess.run(cmd, stderr=subprocess.PIPE)

    if result.returncode != 0:
        error_text = result.stderr.decode(errors="replace")
        logger.error(
            "Run %s failed (returncode=%d):\n%s",
            run["run_id"],
            result.returncode,
            error_text,
        )
        return {
            "model": run["model"],
            "fold": run["fold"],
            "run_id": run["run_id"],
            "success": False,
            "error": error_text,
        }

    # ------------------------------------------------------------------
    # Load output JSON
    # ------------------------------------------------------------------
    if output_json.exists():
        with output_json.open() as fh:
            return json.load(fh)

    logger.error("Run %s succeeded but output JSON not found: %s", run["run_id"], output_json)
    return {
        "run_id": run["run_id"],
        "success": False,
        "error": "output JSON not found after run",
    }


def collect_results(results_dir: Path, models: list[str]) -> pd.DataFrame:
    """Collect per-run result JSONs into a single DataFrame.

    Parameters
    ----------
    results_dir : Path
        Directory to scan for ``*_fold*_results.json`` files.
    models : list[str]
        Only files whose base name starts with one of these model names are
        included.

    Returns
    -------
    pd.DataFrame
        One row per completed run with columns: ``model``, ``fold``,
        ``best_map50``, ``best_map50_95``, ``duration_seconds``,
        ``epochs_trained``, ``error``.  May be empty if no matching files are
        found.

    Notes
    -----
    A WARNING is logged for every expected file that is missing.
    """
    records: list[dict] = []
    matched_files = list(results_dir.glob("*_fold*_results.json"))

    # Filter to files belonging to one of the requested models
    for path in matched_files:
        matched_model = None
        for m in models:
            if path.name.startswith(m):
                matched_model = m
                break
        if matched_model is None:
            continue

        try:
            with path.open() as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("Could not load %s: %s", path, exc)
            continue

        records.append(
            {
                "model": data.get("model", matched_model),
                "fold": data.get("fold", None),
                "best_map50": data.get("best_map50", -1),
                "best_map50_95": data.get("best_map50_95", -1),
                "duration_seconds": data.get("duration_seconds", None),
                "epochs_trained": data.get("epochs_trained", None),
                "error": data.get("error", None),
            }
        )

    if not records:
        logger.warning("No result files found in %s for models %s", results_dir, models)
        return pd.DataFrame(
            columns=[
                "model",
                "fold",
                "best_map50",
                "best_map50_95",
                "duration_seconds",
                "epochs_trained",
                "error",
            ]
        )

    return pd.DataFrame(records)


def compute_ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-fold results by model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``collect_results``.

    Returns
    -------
    pd.DataFrame
        One row per model with columns: ``model``, ``map50_mean``,
        ``map50_std``, ``map50_95_mean``, ``map50_95_std``,
        ``duration_mean``, ``n_folds_completed``.  Returns an empty DataFrame
        with those columns if *df* is empty.

    Notes
    -----
    Rows where ``best_map50 == -1`` (indicating a failed run) are excluded
    from the mean / std calculations.
    """
    expected_cols = [
        "model",
        "map50_mean",
        "map50_std",
        "map50_95_mean",
        "map50_95_std",
        "duration_mean",
        "n_folds_completed",
    ]

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    rows: list[dict] = []
    for model, group in df.groupby("model"):
        valid = group[group["best_map50"] != -1]
        rows.append(
            {
                "model": model,
                "map50_mean": valid["best_map50"].mean() if not valid.empty else float("nan"),
                "map50_std": valid["best_map50"].std() if not valid.empty else float("nan"),
                "map50_95_mean": (
                    valid["best_map50_95"].mean() if not valid.empty else float("nan")
                ),
                "map50_95_std": (
                    valid["best_map50_95"].std() if not valid.empty else float("nan")
                ),
                "duration_mean": group["duration_seconds"].mean(),
                "n_folds_completed": int(len(valid)),
            }
        )

    return pd.DataFrame(rows, columns=expected_cols)


def print_ablation_table(summary_df: pd.DataFrame) -> None:
    """Print a Rich table summarising ablation results and a recommendation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compute_ablation_summary``.

    Returns
    -------
    None

    Notes
    -----
    - The row with the highest ``map50_mean`` is highlighted in green.
    - If *summary_df* is empty a short notice is printed instead.
    """
    console = Console()

    if summary_df.empty:
        console.print("[yellow]No results available yet[/yellow]")
        return

    # Determine best model
    best_idx = summary_df["map50_mean"].idxmax()
    best_model = summary_df.loc[best_idx, "model"]
    best_map50 = summary_df.loc[best_idx, "map50_mean"]

    table = Table(title="Ablation Summary", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("mAP50 Mean", justify="right")
    table.add_column("mAP50 Std", justify="right")
    table.add_column("mAP50-95 Mean", justify="right")
    table.add_column("mAP50-95 Std", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Folds", justify="right")

    for _, row in summary_df.iterrows():
        is_best = row["model"] == best_model
        style = "green" if is_best else ""

        def _fmt(v: float) -> str:
            return f"{v:.4f}" if pd.notna(v) else "N/A"

        def _fmt_dur(v: float) -> str:
            return f"{v:.1f}" if pd.notna(v) else "N/A"

        table.add_row(
            str(row["model"]),
            _fmt(row["map50_mean"]),
            _fmt(row["map50_std"]),
            _fmt(row["map50_95_mean"]),
            _fmt(row["map50_95_std"]),
            _fmt_dur(row["duration_mean"]),
            str(int(row["n_folds_completed"])),
            style=style,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Recommended model: {best_model} (mAP50={best_map50:.4f})[/bold green]"
    )


def main() -> None:
    """Entry point for the ablation training orchestrator.

    Parses CLI arguments, builds the run matrix, optionally executes all
    training runs, collects results, and prints a summary table.

    Returns
    -------
    None
        Exits with code 0 on full success, 1 if any run failed.
    """
    parser = argparse.ArgumentParser(
        description="Ablation training orchestrator for RSNA Spine Classification."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n", "yolo11m"],
        help="Model names to ablate (default: yolo11n yolo11m).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Fold indices (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset YAML.",
    )
    parser.add_argument(
        "--hyp-yaml",
        type=str,
        default="configs/hyp_base.yaml",
        help="Path to hyperparameter YAML.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing fold splits.",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed",
        help="Root directory of processed data.",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default="runs",
        help="Directory for run outputs and result JSONs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Training device (default: cuda).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have an output JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run matrix and exit without training.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build and display run matrix
    # ------------------------------------------------------------------
    matrix = build_run_matrix(args.models, args.folds)

    console = Console()
    matrix_table = Table(title="Run Matrix", show_header=True, header_style="bold magenta")
    matrix_table.add_column("Model")
    matrix_table.add_column("Fold", justify="right")
    matrix_table.add_column("Run ID")
    matrix_table.add_column("Status")

    project_dir = Path(args.project_dir)
    for run in matrix:
        output_json = project_dir / f"{run['run_id']}_results.json"
        if args.resume and output_json.exists():
            status = "[green]done[/green]"
        else:
            status = "[yellow]pending[/yellow]"
        matrix_table.add_row(run["model"], str(run["fold"]), run["run_id"], status)

    console.print(matrix_table)

    if args.dry_run:
        console.print("[bold]Dry-run mode — exiting without training.[/bold]")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Execute runs sequentially
    # ------------------------------------------------------------------
    results: list[dict] = []
    for run in matrix:
        run_result = run_single_training(run, args)
        results.append(run_result)

    # ------------------------------------------------------------------
    # Persist all raw results
    # ------------------------------------------------------------------
    project_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = project_dir / "ablation_all_results.json"
    with all_results_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Saved all results to %s", all_results_path)

    # ------------------------------------------------------------------
    # Collect, summarise, and display
    # ------------------------------------------------------------------
    df = collect_results(project_dir, args.models)
    summary = compute_ablation_summary(df)

    summary_path = project_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    print_ablation_table(summary)

    # ------------------------------------------------------------------
    # Exit code
    # ------------------------------------------------------------------
    n_failed = sum(
        1
        for r in results
        if r.get("success") is False or r.get("best_map50", 0) == -1
    )
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
