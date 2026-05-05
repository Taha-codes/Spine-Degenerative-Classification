"""
RSNA 2024 Lumbar Spine Degenerative Classification — scoring metric utilities.

Provides:
- weighted_log_loss   : sample-weighted multi-class log-loss for one label column
- rsna_score          : full competition scoring over 25 label columns
- predictions_from_yolo : convert raw YOLO detections to the rsna_score prediction format
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: dict[str, int] = {
    "normal_mild": 1,
    "moderate": 1,
    "severe": 2,
}

CONDITIONS: list[str] = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

LEVELS: list[str] = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

LABEL_COLUMNS: list[str] = [
    f"{cond}_{level}" for cond in CONDITIONS for level in LEVELS
]
# 25 total: spinal_canal_stenosis_l1_l2, spinal_canal_stenosis_l2_l3, ...

# Severity index → sample weight (mirrors SEVERITY_WEIGHTS)
_INDEX_TO_WEIGHT: dict[int, int] = {0: 1, 1: 1, 2: 2}

# Inline CLASS_MAP (identical to bbox_utils.CLASS_MAP) — kept here so this
# module has zero intra-package imports as required by spec.
_CLASS_MAP: dict[int, tuple[str, str]] = {
    0:  ("spinal_canal_stenosis",            "normal_mild"),
    1:  ("spinal_canal_stenosis",            "moderate"),
    2:  ("spinal_canal_stenosis",            "severe"),
    3:  ("right_neural_foraminal_narrowing", "normal_mild"),
    4:  ("right_neural_foraminal_narrowing", "moderate"),
    5:  ("right_neural_foraminal_narrowing", "severe"),
    6:  ("left_neural_foraminal_narrowing",  "normal_mild"),
    7:  ("left_neural_foraminal_narrowing",  "moderate"),
    8:  ("left_neural_foraminal_narrowing",  "severe"),
    9:  ("right_subarticular_stenosis",      "normal_mild"),
    10: ("right_subarticular_stenosis",      "moderate"),
    11: ("right_subarticular_stenosis",      "severe"),
    12: ("left_subarticular_stenosis",       "normal_mild"),
    13: ("left_subarticular_stenosis",       "moderate"),
    14: ("left_subarticular_stenosis",       "severe"),
}

# Severity string → class index (0/1/2)
_SEVERITY_INDEX: dict[str, int] = {"normal_mild": 0, "moderate": 1, "severe": 2}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def weighted_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Compute sample-weighted multi-class log-loss for a single label column.

    Parameters
    ----------
    y_true : np.ndarray
        Shape (N,) integer array of true class indices.
        0 = normal_mild, 1 = moderate, 2 = severe.
    y_pred : np.ndarray
        Shape (N, 3) float array of predicted probabilities for each class.
        Rows need not sum to 1; they are clipped to [1e-7, 1-1e-7] before
        the log is applied.
    weights : np.ndarray or None, optional
        Shape (N,) float array of per-sample weights.  If None, weights are
        derived from ``y_true`` using ``SEVERITY_WEIGHTS``:
        class 0 → 1, class 1 → 1, class 2 → 2.

    Returns
    -------
    float
        Scalar weighted log-loss value:
        ``-sum(weight_i * log(y_pred[i, y_true[i]])) / sum(weights)``.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have incompatible shapes, or if
        ``weights`` length does not match ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    if y_pred.ndim != 2 or y_pred.shape != (n, 3):
        raise ValueError(
            f"y_pred must have shape (N, 3), got {y_pred.shape} for N={n}."
        )

    if weights is None:
        weights = np.array([_INDEX_TO_WEIGHT[int(c)] for c in y_true], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError(
                f"weights must have shape ({n},), got {weights.shape}."
            )

    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Select predicted probability of the true class for each sample
    log_probs = np.log(y_pred_clipped[np.arange(n), y_true])

    return float(-np.sum(weights * log_probs) / np.sum(weights))


def rsna_score(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute the full RSNA 2024 competition score over all 25 label columns.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        Columns: ``study_id`` (str/int) + 25 ``LABEL_COLUMNS`` (int 0/1/2).
        Rows where the label is NaN are excluded per column.
    y_pred_df : pd.DataFrame
        Columns: ``study_id`` + 75 prediction columns.  Each label column
        ``lc`` contributes three columns:
        ``f"{lc}_normal_mild"``, ``f"{lc}_moderate"``, ``f"{lc}_severe"``.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"overall"`` (float): mean loss across all 25 label columns.
        - ``"per_condition"`` (dict[str, float]): mean loss per condition
          (average of its 5 level columns).
        - ``"per_column"`` (dict[str, float]): loss for each of the 25 columns.
        - ``"n_samples"`` (int): total (study, column) pairs scored.

    Raises
    ------
    KeyError
        If ``study_id`` is missing from either DataFrame, or if required
        prediction columns are absent.
    """
    merged = pd.merge(y_true_df, y_pred_df, on="study_id", how="inner")

    per_column: dict[str, float] = {}
    n_samples = 0

    for label_col in LABEL_COLUMNS:
        pred_cols = [
            f"{label_col}_normal_mild",
            f"{label_col}_moderate",
            f"{label_col}_severe",
        ]

        # Drop rows where ground truth is NaN
        valid_mask = merged[label_col].notna()
        sub = merged.loc[valid_mask, [label_col] + pred_cols]

        if len(sub) == 0:
            per_column[label_col] = float("nan")
            continue

        y_true = sub[label_col].to_numpy(dtype=np.int64)
        y_pred = sub[pred_cols].to_numpy(dtype=np.float64)

        loss = weighted_log_loss(y_true, y_pred)
        per_column[label_col] = loss
        n_samples += len(sub)

    # Per-condition scores: mean across the 5 levels
    per_condition: dict[str, float] = {}
    for cond in CONDITIONS:
        cond_losses = [
            per_column[f"{cond}_{level}"]
            for level in LEVELS
            if not math.isnan(per_column.get(f"{cond}_{level}", float("nan")))
        ]
        per_condition[cond] = float(np.mean(cond_losses)) if cond_losses else float("nan")

    # Overall: mean across all 25 columns (ignore NaN columns)
    valid_losses = [v for v in per_column.values() if not math.isnan(v)]
    overall = float(np.mean(valid_losses)) if valid_losses else float("nan")

    return {
        "overall": overall,
        "per_condition": per_condition,
        "per_column": per_column,
        "n_samples": n_samples,
    }


def predictions_from_yolo(
    detections: list[list[dict]],
    study_ids: list[str],
) -> pd.DataFrame:
    """
    Convert raw YOLO detections into the prediction DataFrame expected by
    ``rsna_score``.

    Parameters
    ----------
    detections : list[list[dict]]
        One inner list per study.  Each dict must contain:

        - ``"class_id"`` (int): YOLO class ID in [0, 14].
        - ``"confidence"`` (float): detection confidence score in [0, 1].
        - ``"level"`` (str): vertebral level, e.g. ``"l1_l2"``.

    study_ids : list[str]
        Study identifiers, same length as ``detections``.

    Returns
    -------
    pd.DataFrame
        Columns: ``study_id`` + 75 prediction columns
        (``f"{label_col}_normal_mild"``, ``f"{label_col}_moderate"``,
        ``f"{label_col}_severe"`` for each of the 25 ``LABEL_COLUMNS``).
        Probability vectors sum to 1.  When no detections exist for a
        label column, uniform probabilities ``[1/3, 1/3, 1/3]`` are used.

    Raises
    ------
    ValueError
        If ``detections`` and ``study_ids`` have different lengths.
    """
    if len(detections) != len(study_ids):
        raise ValueError(
            f"detections and study_ids must have the same length; "
            f"got {len(detections)} and {len(study_ids)}."
        )

    # Pre-build prediction column names once
    pred_cols: list[str] = []
    for label_col in LABEL_COLUMNS:
        pred_cols.extend([
            f"{label_col}_normal_mild",
            f"{label_col}_moderate",
            f"{label_col}_severe",
        ])

    rows: list[dict] = []

    for study_dets, sid in zip(detections, study_ids):
        row: dict[str, Any] = {"study_id": sid}

        for label_col in LABEL_COLUMNS:
            # Parse condition and level from label_col
            # label_col format: "<condition>_<level>" where level is e.g. "l1_l2"
            # The level always ends with two parts separated by '_'
            # CONDITIONS can themselves contain underscores, so we strip the
            # level suffix (always "lX_lY" = 5 chars + 1 underscore = last two
            # underscore-separated tokens).
            parts = label_col.rsplit("_", 2)
            # parts[-2] + "_" + parts[-1] gives the level
            level = parts[-2] + "_" + parts[-1]
            condition = parts[0]

            # Collect confidences per severity class for this (condition, level)
            probs = [0.0, 0.0, 0.0]  # [normal_mild, moderate, severe]
            for det in study_dets:
                cid = det["class_id"]
                if cid not in _CLASS_MAP:
                    continue
                det_condition, det_severity = _CLASS_MAP[cid]
                if det_condition != condition:
                    continue
                if det.get("level") != level:
                    continue
                sev_idx = _SEVERITY_INDEX.get(det_severity)
                if sev_idx is None:
                    continue
                # Use the highest confidence detection for each severity slot
                conf = float(det["confidence"])
                if conf > probs[sev_idx]:
                    probs[sev_idx] = conf

            total = sum(probs)
            if total == 0.0:
                # No detections for this label_col — fall back to uniform
                probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            else:
                probs = [p / total for p in probs]

            row[f"{label_col}_normal_mild"] = probs[0]
            row[f"{label_col}_moderate"]    = probs[1]
            row[f"{label_col}_severe"]      = probs[2]

        rows.append(row)

    return pd.DataFrame(rows, columns=["study_id"] + pred_cols)


# ---------------------------------------------------------------------------
# Self-test / verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.metrics import log_loss as sklearn_log_loss

    rng = np.random.default_rng(42)
    n_studies = 10
    study_ids = [f"s{i}" for i in range(n_studies)]

    # -----------------------------------------------------------------------
    # 1. Build synthetic y_true_df and y_pred_df
    # -----------------------------------------------------------------------
    true_data: dict[str, Any] = {"study_id": study_ids}
    for col in LABEL_COLUMNS:
        true_data[col] = rng.integers(0, 3, size=n_studies).tolist()
    y_true_df = pd.DataFrame(true_data)

    pred_data: dict[str, Any] = {"study_id": study_ids}
    for col in LABEL_COLUMNS:
        raw = rng.dirichlet(np.ones(3), size=n_studies)
        for k, sev in enumerate(["normal_mild", "moderate", "severe"]):
            pred_data[f"{col}_{sev}"] = raw[:, k].tolist()
    y_pred_df = pd.DataFrame(pred_data)

    # -----------------------------------------------------------------------
    # 2. Call rsna_score and print result
    # -----------------------------------------------------------------------
    result = rsna_score(y_true_df, y_pred_df)
    print("rsna_score result:")
    print(f"  overall    : {result['overall']:.6f}")
    print(f"  n_samples  : {result['n_samples']}")
    print("  per_condition:")
    for cond, loss in result["per_condition"].items():
        print(f"    {cond}: {loss:.6f}")
    print("  per_column (first 5):")
    for col, loss in list(result["per_column"].items())[:5]:
        print(f"    {col}: {loss:.6f}")
    print()

    # -----------------------------------------------------------------------
    # 3. Verify uniform prediction ≈ 1.099  (ln 3 ≈ 1.0986)
    # -----------------------------------------------------------------------
    y_true_uni = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    y_pred_uni = np.full((5, 3), 1.0 / 3.0, dtype=np.float64)
    uni_loss = weighted_log_loss(y_true_uni, y_pred_uni)
    assert abs(uni_loss - 1.099) < 0.01, (
        f"Uniform prediction loss {uni_loss:.6f} not within ±0.01 of 1.099"
    )
    print(f"[OK] Uniform prediction loss = {uni_loss:.6f} (expected ≈ 1.099)")

    # -----------------------------------------------------------------------
    # 4. Verify perfect prediction scores exactly 0.0
    # -----------------------------------------------------------------------
    y_true_perf = np.array([0, 1, 2, 0, 2], dtype=np.int64)
    y_pred_perf = np.eye(3, dtype=np.float64)[y_true_perf]
    perf_loss = weighted_log_loss(y_true_perf, y_pred_perf)
    # Perfect predictions clip to (1 - 1e-7), so loss is very small but
    # technically not exactly 0.0; the spec says "scores exactly 0.0" in the
    # sense of being near-zero.  We verify < 1e-5.
    assert perf_loss < 1e-5, (
        f"Perfect prediction loss {perf_loss} is not near zero"
    )
    print(f"[OK] Perfect prediction loss  = {perf_loss:.2e} (expected ≈ 0.0)")

    # -----------------------------------------------------------------------
    # 5. sklearn comparison
    # -----------------------------------------------------------------------
    y_true_sk = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred_sk = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    sample_weights = np.array(
        [_INDEX_TO_WEIGHT[int(c)] for c in y_true_sk], dtype=np.float64
    )

    our_loss = weighted_log_loss(y_true_sk, y_pred_sk, weights=sample_weights)
    sk_loss = sklearn_log_loss(
        y_true_sk,
        y_pred_sk,
        sample_weight=sample_weights,
        labels=[0, 1, 2],
    )
    assert abs(our_loss - sk_loss) < 1e-6, (
        f"weighted_log_loss ({our_loss:.8f}) differs from sklearn "
        f"({sk_loss:.8f}) by more than 1e-6"
    )
    print(
        f"[OK] sklearn comparison: ours={our_loss:.8f}, sklearn={sk_loss:.8f}, "
        f"diff={abs(our_loss - sk_loss):.2e}"
    )

    print("\nAll checks passed.")
