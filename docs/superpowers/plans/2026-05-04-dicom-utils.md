# Stage 1: dicom_utils.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/utils/dicom_utils.py` — a pure DICOM utility module with zero internal dependencies for the RSNA 2024 Lumbar Spine pipeline.

**Architecture:** Eight public functions covering DICOM loading, pixel extraction, windowing, CLAHE, orientation correction, metadata extraction, and a convenience wrapper. Private helpers `_extract_window_value` and `_minmax_normalize` avoid duplication. The module has no imports from other `src/` modules.

**Tech Stack:** pydicom 3.0.1, numpy 1.26.4, opencv (cv2) 4.10.0, Python 3.10+ (union type hints via `|`)

---

## Constraint
**Only create `src/utils/dicom_utils.py`** (plus necessary empty `__init__.py` files for package structure). Do not create additional source modules.

---

## Key Implementation Decisions

### `_extract_window_value(val) -> float`
WindowCenter/WindowWidth may be `DSfloat` (scalar-like) or `MultiValue` (sequence). Detection: `hasattr(val, "__len__") and not isinstance(val, str)` — covers `MultiValue` without falsely treating `DSfloat` as a sequence (DSfloat has no `__len__`).

### `apply_window=False` skips both windowing AND CLAHE
CLAHE requires uint8 input. If windowing is skipped, the array stays float32 and CLAHE would crash. The flag controls both steps together.

### `fix_orientation` uses `< 0` comparison
Real RSNA data has `-0.0` in `ImageOrientationPatient`. Python's `float(-0.0) < 0` evaluates to `False` (IEEE 754), so the `< 0` comparison correctly does not flip on `-0.0`.

### `dicom_to_numpy` metadata safe defaults
Missing UID tags → `"unknown"`. Missing `InstanceNumber` → `"unknown"` (string, not int, to stay consistent with the string-type DICOM tag).

---

## File Structure

```
src/
├── __init__.py                    (empty)
└── utils/
    ├── __init__.py                (empty)
    └── dicom_utils.py             (implementation — ~280 lines)
tests/
└── utils/
    └── test_dicom_utils.py        (61 tests — written after the file per user constraint)
```

---

## Task 1: Package scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`

- [ ] Create both empty `__init__.py` files

---

## Task 2: Private helpers

**Files:**
- Create: `src/utils/dicom_utils.py`

- [ ] Write module header, imports, logger

```python
"""
DICOM utility functions for the RSNA 2024 Lumbar Spine pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pydicom

logger = logging.getLogger(__name__)
```

- [ ] Write `_extract_window_value`

```python
def _extract_window_value(val) -> float:
    """Extract a scalar float from a DSfloat or MultiValue window tag."""
    if hasattr(val, "__len__") and not isinstance(val, str):
        return float(val[0])
    return float(val)
```

- [ ] Write `_minmax_normalize`

```python
def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 255] uint8 via min/max; returns zeros for flat arrays."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
```

---

## Task 3: `get_pixel_spacing` and `get_series_description`

- [ ] Implement `get_pixel_spacing` — returns `(float, float)`, default `(1.0, 1.0)`, warns on missing tag
- [ ] Implement `get_series_description` — returns stripped lowercase string, default `"unknown"`, warns on missing tag

---

## Task 4: `load_dicom`

- [ ] Implement `load_dicom` — converts path to `Path`, raises `FileNotFoundError` if not exists, calls `pydicom.dcmread(str(path))`

---

## Task 5: `get_pixel_array`

- [ ] Implement `get_pixel_array` — casts to float32, applies `RescaleSlope`/`RescaleIntercept` if both present via `hasattr`

---

## Task 6: `apply_windowing`

- [ ] Implement `apply_windowing` with:
  - kwargs override header tags
  - `_extract_window_value` for DSfloat/MultiValue
  - zero window width → `_minmax_normalize` + warning
  - no tags + no kwargs → `_minmax_normalize` + warning

---

## Task 7: `apply_clahe`

- [ ] Implement `apply_clahe` — `cv2.createCLAHE(clipLimit, tileGridSize).apply(arr)`

---

## Task 8: `fix_orientation`

- [ ] Implement `fix_orientation`:
  - index 0 = row x-component → flip LR if `< 0`
  - index 4 = col y-component → flip UD if `< 0`
  - return unchanged + warn if tag absent

---

## Task 9: `dicom_to_numpy`

- [ ] Implement `dicom_to_numpy` chaining all functions
- [ ] Build metadata dict with 7 required keys

---

## Task 10: `__main__` block

- [ ] Add `if __name__ == "__main__":` block:
  - reads `sys.argv[1]` as DICOM path
  - calls `dicom_to_numpy()`
  - prints metadata dict
  - saves `test_output.png` via `cv2.imwrite`

---

## Verification

```bash
# Quick self-test with a real DICOM from the dataset
python -m src.utils.dicom_utils RSNA/train_images/<study_id>/<series_id>/<instance>.dcm
# Expected: metadata dict printed, test_output.png saved

# Type-check
python -c "import src.utils.dicom_utils; print('imports OK')"
```

### Test Suite (create after main file — 61 tests across 9 classes)
```bash
pytest tests/utils/test_dicom_utils.py -v
# Expected: 61 passed
```
