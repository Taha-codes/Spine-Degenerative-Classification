"""
RSNA 2024 Lumbar Spine Degenerative Classification — Stage 14: Model Comparison.

Loads training and evaluation results for multiple YOLO models across 5-fold
cross-validation, computes summary statistics, generates comparison tables,
and optionally produces plots and a LaTeX table.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.bbox_utils import CLASS_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_all_run_results(runs_dir: Path, models: list[str]) -> pd.DataFrame:
    """
    Load training results JSONs for all models across all folds.

    Parameters
    ----------
    runs_dir : Path
        Directory containing ``{model}_fold{k}_results.json`` files.
    models : list[str]
        List of model names to load (e.g. ``["yolo11n", "yolo11m"]``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, fold, best_map50, best_map50_95,
        duration_seconds, epochs_trained, has_error.  Up to
        ``len(models) * 5`` rows.  Missing JSON files are skipped with a
        WARNING log message.

    Notes
    -----
    Each JSON is expected to contain keys: model, fold, best_map50,
    best_map50_95, duration_seconds, epochs_trained, and optionally error.
    """
    rows: list[dict] = []
    for model in models:
        for k in range(5):
            json_path = runs_dir / f"{model}_fold{k}_results.json"
            if not json_path.exists():
                logger.warning("Missing training results: %s", json_path)
                continue
            try:
                with open(json_path, "r") as fh:
                    data = json.load(fh)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", json_path, exc)
                continue

            rows.append(
                {
                    "model": data.get("model", model),
                    "fold": data.get("fold", k),
                    "best_map50": float(data.get("best_map50", -1.0)),
                    "best_map50_95": float(data.get("best_map50_95", -1.0)),
                    "duration_seconds": float(data.get("duration_seconds", 0.0)),
                    "epochs_trained": int(data.get("epochs_trained", 0)),
                    "has_error": bool(data.get("error", False)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model", "fold", "best_map50", "best_map50_95",
                "duration_seconds", "epochs_trained", "has_error",
            ]
        )
    return pd.DataFrame(rows)


def load_all_eval_results(runs_dir: Path, models: list[str]) -> pd.DataFrame:
    """
    Load evaluation results JSONs for all models across all folds.

    Parameters
    ----------
    runs_dir : Path
        Directory containing ``{model}_fold{k}_eval_results.json`` files.
    models : list[str]
        List of model names to load (e.g. ``["yolo11n", "yolo11m"]``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, fold, map50_overall, map50_95_overall,
        rsna_score_overall, n_val_images, n_val_studies, and one
        ``ap50_{class_name}`` column per entry in CLASS_NAMES (15 total).
        Missing JSON files are skipped with a WARNING log message.

    Notes
    -----
    Each JSON is expected to contain keys: map50_overall, map50_95_overall,
    rsna_score_overall, n_val_images, n_val_studies, per_class_ap50 (dict).
    """
    base_columns = [
        "model", "fold", "map50_overall", "map50_95_overall",
        "rsna_score_overall", "n_val_images", "n_val_studies",
    ]
    ap50_columns = [f"ap50_{cn}" for cn in CLASS_NAMES]

    rows: list[dict] = []
    for model in models:
        for k in range(5):
            json_path = runs_dir / f"{model}_fold{k}_eval_results.json"
            if not json_path.exists():
                logger.warning("Missing eval results: %s", json_path)
                continue
            try:
                with open(json_path, "r") as fh:
                    data = json.load(fh)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", json_path, exc)
                continue

            per_class = data.get("per_class_ap50", {})
            row: dict = {
                "model": model,
                "fold": k,
                "map50_overall": float(data.get("map50_overall", -1.0)),
                "map50_95_overall": float(data.get("map50_95_overall", -1.0)),
                "rsna_score_overall": float(data.get("rsna_score_overall", float("nan"))),
                "n_val_images": int(data.get("n_val_images", 0)),
                "n_val_studies": int(data.get("n_val_studies", 0)),
            }
            for cn in CLASS_NAMES:
                row[f"ap50_{cn}"] = float(per_class.get(cn, -1.0))

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_columns + ap50_columns)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def compute_model_summary(run_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model summary statistics by merging training and eval results.

    Parameters
    ----------
    run_df : pd.DataFrame
        Training results DataFrame as returned by ``load_all_run_results``.
    eval_df : pd.DataFrame
        Evaluation results DataFrame as returned by ``load_all_eval_results``.

    Returns
    -------
    pd.DataFrame
        One row per model with columns: model, map50_mean, map50_std,
        map50_95_mean, map50_95_std, rsna_score_mean, rsna_score_std,
        duration_mean_minutes, n_folds, best_fold.
        Returns an empty DataFrame if both inputs are empty.

    Notes
    -----
    Rows where ``has_error=True`` or ``best_map50=-1`` are excluded from
    aggregation.  ``best_fold`` is the fold with the highest ``best_map50``
    among valid rows.
    """
    if run_df.empty and eval_df.empty:
        return pd.DataFrame(
            columns=[
                "model", "map50_mean", "map50_std", "map50_95_mean", "map50_95_std",
                "rsna_score_mean", "rsna_score_std", "duration_mean_minutes",
                "n_folds", "best_fold",
            ]
        )

    # Merge on (model, fold) with outer join
    if run_df.empty:
        merged = eval_df.copy()
        merged["has_error"] = False
        merged["best_map50"] = -1.0
        merged["best_map50_95"] = -1.0
        merged["duration_seconds"] = 0.0
        merged["epochs_trained"] = 0
    elif eval_df.empty:
        merged = run_df.copy()
        for col in ["map50_overall", "map50_95_overall", "rsna_score_overall",
                    "n_val_images", "n_val_studies"]:
            merged[col] = float("nan")
    else:
        merged = pd.merge(run_df, eval_df, on=["model", "fold"], how="outer")
        if "has_error" not in merged.columns:
            merged["has_error"] = False
        else:
            merged["has_error"] = merged["has_error"].fillna(False)

    # Filter out error rows and sentinel -1 rows
    mask_valid = (~merged["has_error"].astype(bool)) & (merged.get("best_map50", pd.Series(0.0, index=merged.index)) != -1.0)
    valid = merged[mask_valid].copy()

    summary_rows: list[dict] = []
    all_models = merged["model"].dropna().unique().tolist()

    for model in all_models:
        grp = valid[valid["model"] == model]

        if grp.empty:
            summary_rows.append(
                {
                    "model": model,
                    "map50_mean": float("nan"),
                    "map50_std": float("nan"),
                    "map50_95_mean": float("nan"),
                    "map50_95_std": float("nan"),
                    "rsna_score_mean": float("nan"),
                    "rsna_score_std": float("nan"),
                    "duration_mean_minutes": float("nan"),
                    "n_folds": 0,
                    "best_fold": -1,
                }
            )
            continue

        # Prefer map50_overall from eval_df if available, else best_map50 from run_df
        if "map50_overall" in grp.columns and grp["map50_overall"].notna().any():
            map50_vals = grp["map50_overall"].dropna()
        elif "best_map50" in grp.columns:
            map50_vals = grp["best_map50"].dropna()
        else:
            map50_vals = pd.Series(dtype=float)

        if "map50_95_overall" in grp.columns and grp["map50_95_overall"].notna().any():
            map50_95_vals = grp["map50_95_overall"].dropna()
        elif "best_map50_95" in grp.columns:
            map50_95_vals = grp["best_map50_95"].dropna()
        else:
            map50_95_vals = pd.Series(dtype=float)

        rsna_vals = (
            grp["rsna_score_overall"].dropna()
            if "rsna_score_overall" in grp.columns
            else pd.Series(dtype=float)
        )

        dur_vals = (
            grp["duration_seconds"].dropna()
            if "duration_seconds" in grp.columns
            else pd.Series(dtype=float)
        )

        # best_fold: fold with highest best_map50 (from run_df)
        if "best_map50" in grp.columns and grp["best_map50"].notna().any():
            best_idx = grp["best_map50"].idxmax()
            best_fold = int(grp.loc[best_idx, "fold"])
        elif "fold" in grp.columns:
            best_fold = int(grp["fold"].iloc[0])
        else:
            best_fold = -1

        summary_rows.append(
            {
                "model": model,
                "map50_mean": float(map50_vals.mean()) if len(map50_vals) > 0 else float("nan"),
                "map50_std": float(map50_vals.std(ddof=0)) if len(map50_vals) > 1 else 0.0,
                "map50_95_mean": float(map50_95_vals.mean()) if len(map50_95_vals) > 0 else float("nan"),
                "map50_95_std": float(map50_95_vals.std(ddof=0)) if len(map50_95_vals) > 1 else 0.0,
                "rsna_score_mean": float(rsna_vals.mean()) if len(rsna_vals) > 0 else float("nan"),
                "rsna_score_std": float(rsna_vals.std(ddof=0)) if len(rsna_vals) > 1 else 0.0,
                "duration_mean_minutes": float(dur_vals.mean() / 60.0) if len(dur_vals) > 0 else float("nan"),
                "n_folds": len(grp),
                "best_fold": best_fold,
            }
        )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Per-class delta
# ---------------------------------------------------------------------------


def compute_per_class_delta(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class AP50 delta between yolo11m and yolo11n models.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation results DataFrame as returned by ``load_all_eval_results``.
        Must contain ``ap50_{class_name}`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: class_name, yolo11n_ap50, yolo11m_ap50,
        delta (yolo11m − yolo11n), pct_improvement.
        Sorted by delta descending.  Missing models produce zeros for the
        absent model's AP50 and NaN for pct_improvement where applicable.

    Notes
    -----
    ``pct_improvement`` is ``NaN`` when ``yolo11n_ap50 <= 0``.
    """
    def _mean_ap50_for_model(df: pd.DataFrame, model_name: str) -> dict[str, float]:
        subset = df[df["model"] == model_name] if "model" in df.columns else pd.DataFrame()
        result: dict[str, float] = {}
        for cn in CLASS_NAMES:
            col = f"ap50_{cn}"
            if not subset.empty and col in subset.columns:
                vals = subset[col].dropna()
                vals = vals[vals >= 0]  # exclude sentinel -1 values
                result[cn] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                result[cn] = 0.0
        return result

    n_ap50 = _mean_ap50_for_model(eval_df, "yolo11n")
    m_ap50 = _mean_ap50_for_model(eval_df, "yolo11m")

    rows: list[dict] = []
    for cn in CLASS_NAMES:
        n_val = n_ap50[cn]
        m_val = m_ap50[cn]
        delta = m_val - n_val
        pct = (delta / n_val * 100.0) if n_val > 0 else float("nan")
        rows.append(
            {
                "class_name": cn,
                "yolo11n_ap50": n_val,
                "yolo11m_ap50": m_val,
                "delta": delta,
                "pct_improvement": pct,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("delta", ascending=False).reset_index(drop=True)
    return result_df


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def build_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a human-readable model comparison table with formatted metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame as returned by ``compute_model_summary``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Model, mAP@50, mAP@50-95, RSNA Score,
        Train Time (min), Folds, Recommended.
        Each metric is formatted as "mean ± std".
        Returns an empty DataFrame if ``summary_df`` is empty.

    Notes
    -----
    The Recommended column marks "✓" for the model with the lowest
    rsna_score_mean (competition metric, lower is better).  If rsna scores
    are all NaN, the model with the highest map50_mean is recommended instead.
    """
    if summary_df.empty:
        return pd.DataFrame(
            columns=["Model", "mAP@50", "mAP@50-95", "RSNA Score",
                     "Train Time (min)", "Folds", "Recommended"]
        )

    def _fmt(mean_col: str, std_col: str, row: pd.Series, precision: int = 4) -> str:
        mean_val = row.get(mean_col, float("nan"))
        std_val = row.get(std_col, float("nan"))
        if pd.isna(mean_val):
            return "N/A"
        std_str = f"{std_val:.{precision}f}" if not pd.isna(std_val) else "0.0000"
        return f"{mean_val:.{precision}f} ± {std_str}"

    # Determine recommended model
    recommended_model: Optional[str] = None
    rsna_vals = summary_df["rsna_score_mean"].dropna()
    if len(rsna_vals) > 0:
        best_idx = rsna_vals.idxmin()
        recommended_model = str(summary_df.loc[best_idx, "model"])
    else:
        map50_vals = summary_df["map50_mean"].dropna()
        if len(map50_vals) > 0:
            best_idx = map50_vals.idxmax()
            recommended_model = str(summary_df.loc[best_idx, "model"])

    table_rows: list[dict] = []
    for _, row in summary_df.iterrows():
        model_name = str(row["model"])
        dur = row.get("duration_mean_minutes", float("nan"))
        dur_str = f"{dur:.1f}" if not pd.isna(dur) else "N/A"
        table_rows.append(
            {
                "Model": model_name,
                "mAP@50": _fmt("map50_mean", "map50_std", row),
                "mAP@50-95": _fmt("map50_95_mean", "map50_95_std", row),
                "RSNA Score": _fmt("rsna_score_mean", "rsna_score_std", row),
                "Train Time (min)": dur_str,
                "Folds": int(row.get("n_folds", 0)),
                "Recommended": "✓" if model_name == recommended_model else "",
            }
        )

    return pd.DataFrame(table_rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Produce and save two comparison figures to output_dir.

    Figure 1 (model_comparison_bar.png): grouped bar chart comparing mAP@50
    and RSNA Loss across models, with error bars.

    Figure 2 (per_class_delta.png): horizontal bar chart of per-class
    AP50 delta (yolo11m − yolo11n), coloured green/red by sign.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame as returned by ``compute_model_summary``.
    per_class_df : pd.DataFrame
        Per-class delta DataFrame as returned by ``compute_per_class_delta``.
    output_dir : Path
        Directory where PNGs are saved.  Created if it does not exist.

    Returns
    -------
    None

    Notes
    -----
    Figures are saved at 300 DPI.  RSNA Loss Y-axis is inverted so that
    "lower is better" visually reads as "taller bar = better".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: grouped bar chart ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Model Comparison: yolo11n vs yolo11m", fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452"]  # blue, orange
    model_names = summary_df["model"].tolist() if not summary_df.empty else []
    n_models = len(model_names)

    # Subplot 1: mAP@50
    ax1 = axes[0]
    if n_models > 0:
        x = np.arange(n_models)
        bar_width = 0.5
        map50_means = summary_df["map50_mean"].fillna(0).tolist()
        map50_stds = summary_df["map50_std"].fillna(0).tolist()
        for i, (name, mean_val, std_val) in enumerate(zip(model_names, map50_means, map50_stds)):
            ax1.bar(
                i, mean_val, bar_width,
                color=colors[i % len(colors)],
                label=name,
                yerr=std_val,
                capsize=5,
                error_kw={"elinewidth": 1.5},
            )
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, fontsize=9)
    ax1.set_title("mAP@50", fontsize=11)
    ax1.set_ylabel("mAP@50")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot 2: RSNA Loss (inverted Y so lower = taller bar)
    ax2 = axes[1]
    if n_models > 0:
        rsna_means = summary_df["rsna_score_mean"].fillna(0).tolist()
        rsna_stds = summary_df["rsna_score_std"].fillna(0).tolist()
        for i, (name, mean_val, std_val) in enumerate(zip(model_names, rsna_means, rsna_stds)):
            ax2.bar(
                i, mean_val, bar_width,
                color=colors[i % len(colors)],
                label=name,
                yerr=std_val,
                capsize=5,
                error_kw={"elinewidth": 1.5},
            )
        ax2.set_xticks(np.arange(n_models))
        ax2.set_xticklabels(model_names, fontsize=9)
        ax2.invert_yaxis()
    ax2.set_title("RSNA Loss (lower is better)", fontsize=11)
    ax2.set_ylabel("RSNA Score ↓")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.annotate(
        "↑ inverted: higher bar = lower loss = better",
        xy=(0.5, 0.01), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=7, color="gray",
    )

    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model_comparison_bar.png to %s", output_dir)

    # ---- Figure 2: per-class delta horizontal bar chart ----------------------
    if per_class_df.empty:
        logger.warning("per_class_df is empty; skipping per_class_delta.png")
        return

    fig2, ax = plt.subplots(figsize=(10, max(6, len(per_class_df) * 0.45)))

    deltas = per_class_df["delta"].tolist()
    class_labels = per_class_df["class_name"].tolist()
    bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    y_pos = np.arange(len(class_labels))
    ax.barh(y_pos, deltas, color=bar_colors, edgecolor="none", height=0.7)
    ax.axvline(x=0, color="black", linewidth=1.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_xlabel("ΔAP@50 (yolo11m − yolo11n)", fontsize=10)
    ax.set_title("Per-class AP50 Delta: yolo11m vs yolo11n", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="yolo11m better (Δ > 0)"),
        Patch(facecolor="#d62728", label="yolo11n better (Δ < 0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig2.savefig(output_dir / "per_class_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved per_class_delta.png to %s", output_dir)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def export_latex_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """
    Write a LaTeX table of the model comparison to a file.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Formatted comparison DataFrame as returned by ``build_comparison_table``.
    output_path : Path
        File path for the output ``.tex`` file.  Parent directory is created
        if it does not exist.

    Returns
    -------
    None

    Notes
    -----
    The recommended model row (``Recommended == "✓"``) is rendered in bold.
    Column order: Model, mAP@50, mAP@50-95, RSNA Score, Train (min), Folds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_cols = ["Model", "mAP@50", "mAP@50-95", "RSNA Score", "Train Time (min)", "Folds"]
    col_map = {
        "Model": "Model",
        "mAP@50": "mAP@50",
        "mAP@50-95": "mAP@50-95",
        "RSNA Score": "RSNA Score",
        "Train Time (min)": "Train (min)",
        "Folds": "Folds",
    }

    def _escape_latex(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines: list[str] = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model comparison: yolo11n vs yolo11m}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lllllll}",
        r"\hline",
        " & ".join([col_map.get(c, c) for c in header_cols]) + r" & \\",
        r"\hline",
    ]

    if comparison_df.empty:
        lines.append(r"\multicolumn{7}{c}{No results available} \\")
    else:
        for _, row in comparison_df.iterrows():
            is_recommended = str(row.get("Recommended", "")) == "✓"
            cells: list[str] = []
            for col in header_cols:
                val = _escape_latex(row.get(col, ""))
                cells.append(val)
            row_str = " & ".join(cells) + r" & \\"
            if is_recommended:
                # Bold the entire recommended row
                bold_cells = [f"\\textbf{{{c}}}" for c in cells]
                row_str = " & ".join(bold_cells) + r" & \\"
            lines.append(row_str)

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info("LaTeX table saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    CLI entry point for Stage 14: model comparison.

    Loads training and evaluation results for each model/fold, computes
    summary statistics and per-class deltas, prints a rich comparison table,
    and optionally saves plots and a LaTeX table.

    Command-line arguments
    ----------------------
    --runs-dir : str
        Directory containing result JSON files.  Default: ``"runs"``.
    --models : list[str]
        Model names to compare.  Default: ``["yolo11n", "yolo11m"]``.
    --output-dir : str
        Directory for output CSV, PNG, and TEX files.
        Default: ``"runs/comparison"``.
    --no-plots : flag
        Skip generating PNG figures.
    --no-latex : flag
        Skip generating the LaTeX table.
    """
    parser = argparse.ArgumentParser(
        description="Stage 14: compare YOLO models for RSNA 2024 Lumbar Spine."
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Directory containing result JSON files. Default: runs",
    )
    parser.add_argument(
        "--models", nargs="+", default=["yolo11n", "yolo11m"],
        help="Model names to compare. Default: yolo11n yolo11m",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/comparison",
        help="Output directory for CSVs, PNGs, and TEX. Default: runs/comparison",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating PNG figures.",
    )
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Skip generating the LaTeX table.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 & 2: load data
    logger.info("Loading training results from %s ...", runs_dir)
    run_df = load_all_run_results(runs_dir, args.models)
    logger.info("Loaded %d training result rows.", len(run_df))

    logger.info("Loading evaluation results from %s ...", runs_dir)
    eval_df = load_all_eval_results(runs_dir, args.models)
    logger.info("Loaded %d evaluation result rows.", len(eval_df))

    # Step 3: model summary
    logger.info("Computing model summary ...")
    summary_df = compute_model_summary(run_df, eval_df)
    summary_csv = output_dir / "model_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Model summary saved to %s", summary_csv)

    # Step 4: per-class delta
    logger.info("Computing per-class AP50 delta ...")
    per_class_df = compute_per_class_delta(eval_df)
    per_class_csv = output_dir / "per_class_delta.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    logger.info("Per-class delta saved to %s", per_class_csv)

    # Step 5: comparison table
    comp_df = build_comparison_table(summary_df)

    # Step 6: print with rich
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.style import Style

        console = Console()
        rich_table = Table(
            title="Model Comparison Summary",
            show_header=True,
            header_style="bold cyan",
        )
        for col in comp_df.columns:
            rich_table.add_column(str(col), min_width=12)

        for _, row in comp_df.iterrows():
            is_recommended = str(row.get("Recommended", "")) == "✓"
            row_style = Style(color="green", bold=True) if is_recommended else Style()
            rich_table.add_row(*[str(row[c]) for c in comp_df.columns], style=row_style)

        console.print(rich_table)
    except ImportError:
        logger.warning("rich not available; printing plain table.")
        print(comp_df.to_string(index=False))

    # Step 7: plots
    if not args.no_plots:
        logger.info("Generating comparison plots ...")
        plot_metric_comparison(summary_df, per_class_df, output_dir)

    # Step 8: LaTeX
    if not args.no_latex:
        tex_path = output_dir / "model_comparison.tex"
        logger.info("Exporting LaTeX table to %s ...", tex_path)
        export_latex_comparison(comp_df, tex_path)

    logger.info("Stage 14 complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
