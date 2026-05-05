"""
Stage 13 — Per-level detection performance table for RSNA 2024 Lumbar Spine.

Loads detections JSON and train.csv ground truth, computes per-label-column
metrics (precision, recall, AP@50, RSNA weighted log-loss), formats a paper-
ready table, exports LaTeX, and saves a bar chart.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.bbox_utils import CLASS_MAP, CLASS_NAMES  # noqa: F401
from src.utils.rsna_metric import (
    CONDITIONS,
    LABEL_COLUMNS,
    LEVELS,
    weighted_log_loss,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_detections(detections_json: Path) -> pd.DataFrame:
    """
    Load detections from a JSON file and enrich with condition/severity columns.

    Parameters
    ----------
    detections_json : Path
        Path to a JSON file containing a list of detection dicts.  Each dict
        must include at minimum: study_id, series_id, instance_number,
        class_id, confidence, x_center, y_center, width, height.

    Returns
    -------
    pd.DataFrame
        Original fields plus two derived columns:

        - ``condition`` (str): anatomical condition from CLASS_MAP.
        - ``severity``  (str): severity label from CLASS_MAP.
    """
    with open(detections_json, "r") as fh:
        records: list[dict] = json.load(fh)

    df = pd.DataFrame(records)
    df["condition"] = df["class_id"].map(lambda cid: CLASS_MAP[int(cid)][0])
    df["severity"] = df["class_id"].map(lambda cid: CLASS_MAP[int(cid)][1])
    return df


def load_ground_truth_long(train_csv_path: Path) -> pd.DataFrame:
    """
    Load train.csv (wide format) and return a long-format ground-truth table.

    Parameters
    ----------
    train_csv_path : Path
        Path to the RSNA train.csv file.  Expected wide format with columns
        ``study_id`` and one column per LABEL_COLUMN (25 total).

    Returns
    -------
    pd.DataFrame
        Columns: study_id (str), label_col, condition, level, severity_int.
        Rows where severity_str is NaN are dropped.

    Notes
    -----
    Column names are normalised (lowercase, spaces→underscores, /→_, strip)
    before melting.  Severity strings are mapped case-insensitively:
    ``normal/mild`` → 0, ``moderate`` → 1, ``severe`` → 2.
    """
    wide = pd.read_csv(train_csv_path)

    # Normalise column names
    wide.columns = (
        wide.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # Melt to long format
    long = wide.melt(
        id_vars="study_id",
        value_vars=LABEL_COLUMNS,
        var_name="label_col",
        value_name="severity_str",
    )

    # Drop rows with missing severity
    long = long.dropna(subset=["severity_str"])

    # Map severity string → int (case-insensitive)
    sev_map = {"normal/mild": 0, "moderate": 1, "severe": 2}
    long["severity_int"] = (
        long["severity_str"]
        .str.strip()
        .str.lower()
        .map(sev_map)
    )

    # Parse condition and level from label_col
    def _parse_condition_level(label_col: str) -> tuple[str, str]:
        parts = label_col.split("_")
        level = parts[-2] + "_" + parts[-1]
        condition = "_".join(parts[:-2])
        return condition, level

    parsed = long["label_col"].map(_parse_condition_level)
    long["condition"] = parsed.map(lambda t: t[0])
    long["level"] = parsed.map(lambda t: t[1])

    long["study_id"] = long["study_id"].astype(str)

    return long[["study_id", "label_col", "condition", "level", "severity_int"]]


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _compute_ap50(y_scores: np.ndarray, y_true_binary: np.ndarray) -> float:
    """
    Compute AP@50 via trapezoid integration over a precision-recall curve.

    Parameters
    ----------
    y_scores : np.ndarray
        Shape (N,) float detection confidence scores.
    y_true_binary : np.ndarray
        Shape (N,) int binary ground-truth labels (1 = positive).

    Returns
    -------
    float
        Area under the precision-recall curve, in [0, 1].
    """
    if y_true_binary.sum() == 0:
        return 0.0
    sorted_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true_binary[sorted_idx]
    n_pos = y_true_sorted.sum()
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    # prepend (0, 0) for proper integration
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

_CONF_THRESH = 0.25
_UNIFORM_PRED = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])


def compute_per_column_metrics(
    detections_df: pd.DataFrame,
    gt_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute detection performance metrics for each of the 25 LABEL_COLUMNS.

    Parameters
    ----------
    detections_df : pd.DataFrame
        Output of :func:`load_detections`.  Must include columns:
        study_id, condition, confidence.
    gt_long_df : pd.DataFrame
        Output of :func:`load_ground_truth_long`.  Must include columns:
        study_id, label_col, condition, level, severity_int.

    Returns
    -------
    pd.DataFrame
        25 rows, one per LABEL_COLUMN.  Columns: label_col, condition, level,
        n_normal_mild, n_moderate, n_severe, n_total, precision, recall,
        ap50, rsna_loss.
    """
    rows: list[dict[str, Any]] = []

    # Pre-index detections by condition for speed
    det_has_condition: dict[str, pd.DataFrame] = {}
    if len(detections_df) > 0:
        for cond, grp in detections_df.groupby("condition"):
            det_has_condition[cond] = grp

    for label_col in LABEL_COLUMNS:
        parts = label_col.split("_")
        level = parts[-2] + "_" + parts[-1]
        condition = "_".join(parts[:-2])

        # Ground-truth subset for this label column
        gt_sub = gt_long_df[gt_long_df["label_col"] == label_col].copy()
        gt_sub["study_id"] = gt_sub["study_id"].astype(str)

        n_normal_mild = int((gt_sub["severity_int"] == 0).sum())
        n_moderate = int((gt_sub["severity_int"] == 1).sum())
        n_severe = int((gt_sub["severity_int"] == 2).sum())
        n_total = len(gt_sub)

        gt_study_ids = set(gt_sub["study_id"].tolist())

        cond_dets = det_has_condition.get(condition, pd.DataFrame())
        has_dets = len(cond_dets) > 0

        if not has_dets:
            log.warning(
                "No detections found for condition '%s' (label_col: %s). "
                "Filling precision=0, recall=0, ap50=0, rsna_loss=nan.",
                condition,
                label_col,
            )
            rows.append(
                {
                    "label_col": label_col,
                    "condition": condition,
                    "level": level,
                    "n_normal_mild": n_normal_mild,
                    "n_moderate": n_moderate,
                    "n_severe": n_severe,
                    "n_total": n_total,
                    "precision": 0.0,
                    "recall": 0.0,
                    "ap50": 0.0,
                    "rsna_loss": float("nan"),
                }
            )
            continue

        # High-confidence detections subset
        cond_dets_conf = cond_dets[cond_dets["confidence"] >= _CONF_THRESH].copy()
        cond_dets_conf["study_id"] = cond_dets_conf["study_id"].astype(str)
        detected_studies = set(cond_dets_conf["study_id"].tolist())

        # TP / FN per GT study
        tp = sum(1 for sid in gt_study_ids if sid in detected_studies)
        fn = sum(1 for sid in gt_study_ids if sid not in detected_studies)

        # FP = detected studies not in GT
        fp = sum(1 for sid in detected_studies if sid not in gt_study_ids)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # AP50: use max confidence per study (0.0 if no detection)
        all_det_studies = cond_dets.copy()
        all_det_studies["study_id"] = all_det_studies["study_id"].astype(str)

        max_conf_per_study = (
            all_det_studies.groupby("study_id")["confidence"].max()
        )

        gt_studies_list = list(gt_study_ids)
        y_scores = np.array(
            [float(max_conf_per_study.get(sid, 0.0)) for sid in gt_studies_list],
            dtype=np.float64,
        )
        # All GT samples are "positive" (severity_int >= 0 always true)
        y_true_binary = np.ones(len(gt_studies_list), dtype=np.int32)
        ap50 = _compute_ap50(y_scores, y_true_binary)

        # RSNA loss: build y_true and y_pred arrays for GT studies
        rsna_loss = float("nan")
        if len(gt_sub) > 0:
            y_true_arr = gt_sub.set_index("study_id")["severity_int"]
            y_pred_rows: list[list[float]] = []

            for sid in gt_studies_list:
                study_dets_for_study = all_det_studies[
                    all_det_studies["study_id"] == sid
                ]
                if len(study_dets_for_study) == 0:
                    # Uniform prediction
                    y_pred_rows.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
                else:
                    # Build soft probabilities from detection confidences
                    conf_by_sev = [0.0, 0.0, 0.0]
                    sev_map_local = {"normal_mild": 0, "moderate": 1, "severe": 2}
                    for _, det_row in study_dets_for_study.iterrows():
                        sev_str = CLASS_MAP[int(det_row["class_id"])][1]
                        sev_idx = sev_map_local.get(sev_str, -1)
                        if sev_idx >= 0:
                            conf = float(det_row["confidence"])
                            if conf > conf_by_sev[sev_idx]:
                                conf_by_sev[sev_idx] = conf
                    total_conf = sum(conf_by_sev)
                    if total_conf == 0.0:
                        y_pred_rows.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
                    else:
                        y_pred_rows.append([c / total_conf for c in conf_by_sev])

            y_true_vals = np.array(
                [int(y_true_arr.loc[sid]) for sid in gt_studies_list],
                dtype=np.int64,
            )
            y_pred_vals = np.array(y_pred_rows, dtype=np.float64)
            try:
                rsna_loss = weighted_log_loss(y_true_vals, y_pred_vals)
            except Exception as exc:
                log.warning("weighted_log_loss failed for %s: %s", label_col, exc)
                rsna_loss = float("nan")

        rows.append(
            {
                "label_col": label_col,
                "condition": condition,
                "level": level,
                "n_normal_mild": n_normal_mild,
                "n_moderate": n_moderate,
                "n_severe": n_severe,
                "n_total": n_total,
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "rsna_loss": rsna_loss,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paper table
# ---------------------------------------------------------------------------


def build_paper_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a publication-ready table with per-condition summaries and an
    overall row.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`compute_per_column_metrics` (25 rows).

    Returns
    -------
    pd.DataFrame
        30 rows: 25 data rows + 5 condition-mean rows + 1 overall row.
        Columns: condition, level, n_total, n_severe, precision, recall,
        ap50, rsna_loss.  Float columns rounded to 3 decimal places.
    """
    # Build ordering maps
    cond_order = {c: i for i, c in enumerate(CONDITIONS)}
    level_order = {lv: i for i, lv in enumerate(LEVELS)}

    df = metrics_df.copy()
    df["n_total"] = df["n_normal_mild"] + df["n_moderate"] + df["n_severe"]

    # Sort by condition then level
    df["_cond_idx"] = df["condition"].map(cond_order)
    df["_level_idx"] = df["level"].map(level_order)
    df = df.sort_values(["_cond_idx", "_level_idx"]).drop(
        columns=["_cond_idx", "_level_idx"]
    )

    float_cols = ["precision", "recall", "ap50", "rsna_loss"]
    output_cols = ["condition", "level", "n_total", "n_severe", "precision", "recall", "ap50", "rsna_loss"]

    result_rows: list[dict[str, Any]] = []

    # Add data rows + per-condition mean rows
    for cond in CONDITIONS:
        cond_rows = df[df["condition"] == cond]
        for _, row in cond_rows.iterrows():
            result_rows.append(
                {
                    "condition": row["condition"],
                    "level": row["level"],
                    "n_total": int(row["n_total"]),
                    "n_severe": int(row["n_severe"]),
                    "precision": round(float(row["precision"]), 3),
                    "recall": round(float(row["recall"]), 3),
                    "ap50": round(float(row["ap50"]), 3),
                    "rsna_loss": round(float(row["rsna_loss"]), 3)
                    if not np.isnan(row["rsna_loss"])
                    else float("nan"),
                }
            )
        # Condition mean row
        mean_row: dict[str, Any] = {
            "condition": cond,
            "level": "MEAN",
            "n_total": int(cond_rows["n_total"].sum()),
            "n_severe": int(cond_rows["n_severe"].sum()),
        }
        for fc in float_cols:
            vals = cond_rows[fc].dropna()
            mean_row[fc] = round(float(vals.mean()), 3) if len(vals) > 0 else float("nan")
        result_rows.append(mean_row)

    # Overall mean row
    overall_row: dict[str, Any] = {
        "condition": "OVERALL",
        "level": "MEAN",
        "n_total": int(df["n_total"].sum()),
        "n_severe": int(df["n_severe"].sum()),
    }
    for fc in float_cols:
        vals = df[fc].dropna()
        overall_row[fc] = round(float(vals.mean()), 3) if len(vals) > 0 else float("nan")
    result_rows.append(overall_row)

    return pd.DataFrame(result_rows, columns=output_cols)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def export_latex_table(paper_df: pd.DataFrame, output_path: Path) -> None:
    """
    Write a LaTeX tabular for the per-level detection results.

    Parameters
    ----------
    paper_df : pd.DataFrame
        Output of :func:`build_paper_table`.
    output_path : Path
        Destination ``.tex`` file.

    Notes
    -----
    - MEAN/OVERALL rows are omitted from the table body.
    - The row with the lowest ``rsna_loss`` per condition group is bolded.
    - \\hline separators are inserted between condition groups.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only data rows (not MEAN/OVERALL)
    data_rows = paper_df[paper_df["level"] != "MEAN"].copy()

    lines: list[str] = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-level detection performance on RSNA 2024 Lumbar Spine}",
        r"\label{tab:per_level_results}",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Condition & Level & N & N\_Severe & Precision & Recall & AP@50 & RSNA Loss \\",
        r"\hline",
    ]

    # Precompute best (lowest rsna_loss) row index per condition
    best_idx_per_cond: dict[str, int] = {}
    for cond in CONDITIONS:
        cond_data = data_rows[data_rows["condition"] == cond]
        valid = cond_data.dropna(subset=["rsna_loss"])
        if len(valid) > 0:
            best_idx_per_cond[cond] = int(valid["rsna_loss"].idxmin())

    prev_cond = None
    for idx, row in data_rows.iterrows():
        cond = row["condition"]

        # \hline between condition groups
        if prev_cond is not None and cond != prev_cond:
            lines.append(r"\hline")
        prev_cond = cond

        is_best = best_idx_per_cond.get(cond) == idx
        rsna_str = f"{row['rsna_loss']:.3f}" if not _is_nan(row["rsna_loss"]) else "—"
        prec_str = f"{row['precision']:.3f}"
        rec_str = f"{row['recall']:.3f}"
        ap_str = f"{row['ap50']:.3f}"

        if is_best:
            rsna_str = rf"\textbf{{{rsna_str}}}"
            prec_str = rf"\textbf{{{prec_str}}}"
            rec_str = rf"\textbf{{{rec_str}}}"
            ap_str = rf"\textbf{{{ap_str}}}"

        cond_clean = cond.replace("_", r"\_")
        level_clean = row["level"].replace("_", r"\_")

        lines.append(
            f"{cond_clean} & {level_clean} & {row['n_total']} & {row['n_severe']} & "
            f"{prec_str} & {rec_str} & {ap_str} & {rsna_str} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    log.info("LaTeX table written to %s", output_path)


def _is_nan(v: Any) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------


def plot_per_condition_bar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a grouped bar chart of AP@50 by level, with one bar per condition.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`compute_per_column_metrics` (25 rows, no MEAN rows).
    output_path : Path
        Destination image file (PNG recommended).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    x = np.arange(len(LEVELS))
    n_cond = len(CONDITIONS)
    bar_width = 0.15
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * bar_width

    fig, ax = plt.subplots(figsize=(12, 5))

    for ci, (cond, color, offset) in enumerate(zip(CONDITIONS, colors, offsets)):
        cond_rows = metrics_df[metrics_df["condition"] == cond]
        ap_vals = [
            float(cond_rows[cond_rows["level"] == lv]["ap50"].iloc[0])
            if len(cond_rows[cond_rows["level"] == lv]) > 0
            else 0.0
            for lv in LEVELS
        ]
        ax.bar(
            x + offset,
            ap_vals,
            width=bar_width,
            label=cond.replace("_", " "),
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([lv.replace("_", "/") for lv in LEVELS])
    ax.set_xlabel("Level")
    ax.set_ylabel("AP@50")
    ax.set_title("Per-condition AP@50 by vertebral level")
    ax.legend(loc="upper right", fontsize=8)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Gridlines on Y only
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log.info("Bar chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# Rich table printer
# ---------------------------------------------------------------------------


def _print_rich_table(paper_df: pd.DataFrame) -> None:
    """Print paper_df using rich.table.Table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Per-level Detection Performance", show_lines=True)

        for col in paper_df.columns:
            table.add_column(col, justify="right" if col not in ("condition", "level") else "left")

        for _, row in paper_df.iterrows():
            style = ""
            if row.get("level") == "MEAN":
                style = "bold"
            cells = []
            for col in paper_df.columns:
                v = row[col]
                if isinstance(v, float) and np.isnan(v):
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            table.add_row(*cells, style=style)

        console.print(table)
    except ImportError:
        log.warning("rich not available; falling back to plain print.")
        print(paper_df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for Stage 13 evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 13: per-level detection performance table."
    )
    parser.add_argument(
        "--detections-json",
        required=True,
        type=Path,
        help="Path to detections JSON file.",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        type=Path,
        help="Path to RSNA train.csv (wide format).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/eval",
        type=Path,
        help="Directory to write outputs (default: runs/eval).",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip LaTeX table export.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip bar chart generation.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    log.info("Loading detections from %s", args.detections_json)
    detections_df = load_detections(args.detections_json)
    log.info("Loaded %d detections", len(detections_df))

    log.info("Loading ground truth from %s", args.train_csv)
    gt_long_df = load_ground_truth_long(args.train_csv)
    log.info("Ground truth: %d rows", len(gt_long_df))

    # 2. Compute metrics
    log.info("Computing per-column metrics …")
    metrics_df = compute_per_column_metrics(detections_df, gt_long_df)

    # 3. Save raw metrics
    metrics_path = output_dir / "per_level_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Saved metrics to %s", metrics_path)

    # 4. Build paper table
    paper_df = build_paper_table(metrics_df)

    # 5. Save paper table
    paper_path = output_dir / "paper_table.csv"
    paper_df.to_csv(paper_path, index=False)
    log.info("Saved paper table to %s", paper_path)

    # 6. Print with rich
    _print_rich_table(paper_df)

    # 7. LaTeX export
    if not args.no_latex:
        latex_path = output_dir / "paper_table.tex"
        export_latex_table(paper_df, latex_path)

    # 8. Bar chart
    if not args.no_plot:
        plot_path = output_dir / "per_condition_ap50.png"
        plot_per_condition_bar(metrics_df, plot_path)


if __name__ == "__main__":
    main()
