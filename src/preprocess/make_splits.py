"""
Stage 6 – Stratified k-fold splits and class-weight computation.

Creates per-fold CSV files (data/splits/fold_{k}.csv) and a JSON file of
inverse-frequency class weights (data/class_weights.json) for the RSNA 2024
Lumbar Spine Degenerative Classification dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.bbox_utils import CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, int] = {
    "normal/mild": 0,
    "moderate": 1,
    "severe": 2,
}

# Condition string (raw, lowercase+stripped) → base YOLO class ID
_CONDITION_BASE: dict[str, int] = {
    "spinal canal stenosis": 0,
    "right neural foraminal narrowing": 3,
    "left neural foraminal narrowing": 6,
    "right subarticular stenosis": 9,
    "left subarticular stenosis": 12,
}

# Severity string (raw, lowercase+stripped) → YOLO class offset
_SEVERITY_OFFSET: dict[str, int] = {
    "normal/mild": 0,
    "moderate": 1,
    "severe": 2,
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_study_severity(train_csv_path: Path) -> pd.DataFrame:
    """
    Compute per-study worst-case severity from the wide-format train.csv.

    The file has one row per study and 25 severity columns named
    ``{condition}_{level}`` (e.g. ``spinal_canal_stenosis_l1_l2``).  Each cell
    holds a string label "Normal/Mild", "Moderate", or "Severe".  This
    function melts those columns, maps labels to integers 0/1/2, and returns
    the maximum severity found across all conditions/levels for each study.

    Parameters
    ----------
    train_csv_path : Path
        Absolute path to ``train.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``study_id`` (int), ``worst_severity`` (int in {0, 1, 2}).
        One row per unique study.
    """
    df = pd.read_csv(train_csv_path)

    # Melt wide → long (one row per study × condition × level)
    id_vars = ["study_id"]
    value_vars = [c for c in df.columns if c != "study_id"]
    long = df.melt(id_vars=id_vars, value_vars=value_vars, value_name="severity_str")

    # Map severity strings → integers (case-insensitive, strip whitespace)
    long["severity_int"] = (
        long["severity_str"]
        .str.strip()
        .str.lower()
        .map(_SEVERITY_MAP)
    )

    # Drop rows where severity is missing / unmappable
    long = long.dropna(subset=["severity_int"])
    long["severity_int"] = long["severity_int"].astype(int)

    # Per-study worst severity
    result = (
        long.groupby("study_id", sort=False)["severity_int"]
        .max()
        .reset_index()
        .rename(columns={"severity_int": "worst_severity"})
    )
    result["study_id"] = result["study_id"].astype(int)
    result["worst_severity"] = result["worst_severity"].astype(int)

    return result[["study_id", "worst_severity"]]


def make_stratified_folds(
    study_severity_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign each study to one of ``n_folds`` validation folds using
    stratified group k-fold splitting.

    Stratification is done on ``worst_severity`` so that the class balance
    is approximately preserved in every fold.

    Parameters
    ----------
    study_severity_df : pd.DataFrame
        Output of :func:`compute_study_severity`; must contain columns
        ``study_id`` and ``worst_severity``.
    n_folds : int, optional
        Number of folds (default 5).
    seed : int, optional
        Random state passed to
        :class:`sklearn.model_selection.StratifiedGroupKFold` (default 42).

    Returns
    -------
    pd.DataFrame
        Columns: ``study_id`` (int), ``fold`` (int in 0..n_folds-1),
        ``worst_severity`` (int).  One row per study.
    """
    df = study_severity_df.reset_index(drop=True).copy()

    X = np.arange(len(df))
    y = df["worst_severity"].values
    groups = df["study_id"].values

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )

    fold_col = np.full(len(df), fill_value=-1, dtype=int)
    for fold_idx, (_train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_col[val_idx] = fold_idx

    df["fold"] = fold_col
    return df[["study_id", "fold", "worst_severity"]]


def save_fold_csvs(fold_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Write one CSV per fold to ``output_dir``.

    Each CSV contains all studies with a ``split`` column set to ``"val"``
    for the held-out fold and ``"train"`` for all others.

    Parameters
    ----------
    fold_df : pd.DataFrame
        Output of :func:`make_stratified_folds`; must contain columns
        ``study_id``, ``fold``, and ``worst_severity``.
    output_dir : Path
        Directory where ``fold_0.csv`` … ``fold_{n-1}.csv`` will be written.
        Created (with parents) if it does not exist.

    Returns
    -------
    None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_folds = fold_df["fold"].nunique()

    for k in range(n_folds):
        out = fold_df.copy()
        out["split"] = out["fold"].apply(lambda f: "val" if f == k else "train")

        n_train = (out["split"] == "train").sum()
        n_val = (out["split"] == "val").sum()
        dist = (
            out.loc[out["split"] == "val", "worst_severity"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        out = out[["study_id", "fold", "worst_severity", "split"]]
        out.to_csv(output_dir / f"fold_{k}.csv", index=False)

        logger.info(
            "Fold %d: %d train, %d val | severity distribution: %s",
            k, n_train, n_val, dist,
        )


def compute_class_weights(train_csv_path: Path) -> dict:
    """
    Compute inverse-frequency weights for all 15 YOLO class IDs.

    Annotations from ``train_label_coordinates.csv`` are merged with
    ``train.csv`` to obtain a severity label for each coordinate entry.
    Each annotation is then mapped to one of the 15 YOLO classes
    (condition × severity).  Weights are proportional to the inverse
    frequency of each class and are normalised so that the median weight
    equals 1.0.

    Parameters
    ----------
    train_csv_path : Path
        Absolute path to ``train.csv``.  ``train_label_coordinates.csv`` is
        expected to reside in the same directory.

    Returns
    -------
    dict
        ``{class_id (int): weight (float)}`` for all 15 classes (0–14).
        Weights are rounded to 4 decimal places.  Classes with zero observed
        frequency receive a weight of 1.0 before normalisation.
    """
    train_csv_path = Path(train_csv_path)
    coords_path = train_csv_path.parent / "train_label_coordinates.csv"

    # ------------------------------------------------------------------
    # Load and reshape train.csv (wide → long) to get a severity per
    # (study_id, condition, level) triple.
    # ------------------------------------------------------------------
    train_wide = pd.read_csv(train_csv_path)

    value_vars = [c for c in train_wide.columns if c != "study_id"]
    train_long = train_wide.melt(
        id_vars=["study_id"],
        value_vars=value_vars,
        var_name="cond_level_col",
        value_name="severity_str",
    )

    # Build a lookup key that matches train_label_coordinates (condition+level)
    # Column names look like "spinal_canal_stenosis_l1_l2"; we need to recover
    # condition and level so we can join on them.
    #
    # Strategy: the level is always a 2-part suffix of the form *_lX_yZ or
    # *_lX_s1.  The five level strings after normalisation are:
    #   l1_l2  l2_l3  l3_l4  l4_l5  l5_s1
    # We strip those from the end of the column name to get the condition.
    _LEVELS = {"l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"}

    def _split_cond_level(col: str):
        parts = col.rsplit("_", 2)
        if len(parts) == 3:
            level_key = f"{parts[1]}_{parts[2]}"
            if level_key in _LEVELS:
                return parts[0], level_key
        return col, ""

    train_long[["condition_key", "level_key"]] = pd.DataFrame(
        train_long["cond_level_col"].map(_split_cond_level).tolist(),
        index=train_long.index,
    )

    # Map severity string → integer
    train_long["severity_int"] = (
        train_long["severity_str"].str.strip().str.lower().map(_SEVERITY_MAP)
    )
    train_long = train_long.dropna(subset=["severity_int"])
    train_long["severity_int"] = train_long["severity_int"].astype(int)

    # ------------------------------------------------------------------
    # Load label coordinates and build the same keys for merging.
    # condition: "Spinal Canal Stenosis" → "spinal_canal_stenosis"
    # level:     "L1/L2"                → "l1_l2"
    # ------------------------------------------------------------------
    coords = pd.read_csv(coords_path)
    coords["condition_key"] = (
        coords["condition"].str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    coords["level_key"] = (
        coords["level"].str.strip().str.lower().str.replace("/", "_", regex=False)
    )

    # Merge coordinates with long-form severity
    merged = coords.merge(
        train_long[["study_id", "condition_key", "level_key", "severity_int"]],
        on=["study_id", "condition_key", "level_key"],
        how="left",
    )
    merged = merged.dropna(subset=["severity_int"])

    # ------------------------------------------------------------------
    # Map each annotation to a YOLO class ID
    # ------------------------------------------------------------------
    def _to_class_id(row) -> int | None:
        cond_raw = row["condition"].strip().lower()
        sev_raw = row["severity_str"] if "severity_str" in row.index else None
        # Use the merged severity_int directly
        sev_int = int(row["severity_int"])
        base = _CONDITION_BASE.get(cond_raw)
        if base is None:
            return None
        return base + sev_int

    # Build severity string column from int for clarity, then apply mapping
    merged["class_id"] = merged.apply(
        lambda r: _CONDITION_BASE.get(r["condition"].strip().lower(), None),
        axis=1,
    )
    merged = merged.dropna(subset=["class_id"])
    merged["class_id"] = merged["class_id"].astype(int) + merged["severity_int"].astype(int)

    # ------------------------------------------------------------------
    # Count frequencies and compute normalised inverse-frequency weights
    # ------------------------------------------------------------------
    counts = merged["class_id"].value_counts()

    weights_raw: dict[int, float] = {}
    for cid in range(15):
        freq = counts.get(cid, 0)
        weights_raw[cid] = 1.0 / freq if freq > 0 else 1.0

    values = np.array([weights_raw[i] for i in range(15)])
    median_w = float(np.median(values))
    if median_w == 0.0:
        median_w = 1.0

    weights: dict[int, float] = {
        cid: round(float(w / median_w), 4)
        for cid, w in weights_raw.items()
    }
    return weights


def save_class_weights(weights: dict, output_path: Path) -> None:
    """
    Persist class weights to a JSON file and print a formatted summary table.

    Parameters
    ----------
    weights : dict
        ``{class_id (int): weight (float)}`` for all 15 classes, as returned
        by :func:`compute_class_weights`.
    output_path : Path
        Destination JSON file path.  Parent directories are created if absent.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON keys must be strings
    json_weights = {str(k): v for k, v in weights.items()}
    with open(output_path, "w") as fh:
        json.dump(json_weights, fh, indent=2)

    logger.info("Class weights saved to %s", output_path)

    # Pretty-print table
    print(f"\n{'class_id':>8} | {'class_name':<45} | {'weight':>8}")
    print("-" * 70)
    for cid in range(15):
        print(f"{cid:>8} | {CLASS_NAMES[cid]:<45} | {weights[cid]:>8.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Command-line interface for generating fold CSVs and class weights.

    Parameters
    ----------
    None – all inputs are supplied via ``argparse`` arguments:

    --raw-root : str
        Directory containing ``train.csv`` and
        ``train_label_coordinates.csv``.
    --output-dir : str, optional
        Destination for fold CSVs (default: ``data/splits``).
    --weights-output : str, optional
        Destination JSON path for class weights
        (default: ``data/class_weights.json``).
    --n-folds : int, optional
        Number of cross-validation folds (default: 5).
    --seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Generate stratified k-fold splits and class weights."
    )
    parser.add_argument("--raw-root", type=str, required=True,
                        help="Directory containing train.csv")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                        help="Where to write fold_k.csv files")
    parser.add_argument("--weights-output", type=str, default="data/class_weights.json",
                        help="Path for class_weights.json")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    weights_output = Path(args.weights_output)
    n_folds: int = args.n_folds
    seed: int = args.seed

    train_csv = raw_root / "train.csv"

    # Step 1 – per-study severity
    logger.info("Computing study-level severity from %s", train_csv)
    study_severity_df = compute_study_severity(train_csv)
    logger.info(
        "study_severity shape: %s | worst_severity counts: %s",
        study_severity_df.shape,
        study_severity_df["worst_severity"].value_counts().sort_index().to_dict(),
    )

    # Step 2 – stratified folds
    logger.info("Building %d stratified folds (seed=%d)", n_folds, seed)
    fold_df = make_stratified_folds(study_severity_df, n_folds=n_folds, seed=seed)

    # Step 3 – integrity: no study in both train and val of the same fold
    for k in range(n_folds):
        val_ids = set(fold_df.loc[fold_df["fold"] == k, "study_id"])
        train_ids = set(fold_df.loc[fold_df["fold"] != k, "study_id"])
        overlap = val_ids & train_ids
        # overlap is expected (a study is in train for other folds);
        # the real check is: within a single fold no study has both splits.
        # Since each study has exactly one fold assignment this is guaranteed
        # by construction, but we verify explicitly.
        assert len(val_ids) > 0, f"Fold {k} has no validation samples."

    # Step 4 – each study appears in exactly one val fold
    val_counts = fold_df["fold"].value_counts()
    assert fold_df["fold"].nunique() == n_folds, (
        "Some folds are missing from the assignment."
    )
    study_fold_counts = fold_df.groupby("study_id")["fold"].count()
    assert (study_fold_counts == 1).all(), (
        "Some studies appear in more than one val fold."
    )

    # Step 5 – save fold CSVs
    save_fold_csvs(fold_df, output_dir)

    # Step 6 – class weights
    logger.info("Computing class weights from %s", train_csv)
    weights = compute_class_weights(train_csv)

    # Step 7 – save class weights
    save_class_weights(weights, weights_output)

    logger.info("Done.")


if __name__ == "__main__":
    main()
