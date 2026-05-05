# Stage 2: bbox_utils.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/utils/bbox_utils.py` — a pure-Python bounding-box utility module with anatomically-correct per-condition YOLO bbox sizes and bidirectional coordinate conversion for the RSNA 2024 Lumbar Spine pipeline.

**Architecture:** Three module-level constants (`BBOX_SIZES`, `CLASS_MAP`, `CLASS_NAMES`) define the ground truth. Four lookup functions provide case-insensitive access. Two coordinate-conversion functions handle pixel↔YOLO transforms. A pre-built `_CLASS_ID_LOOKUP` reverse-index (built once at import time from `CLASS_MAP`) makes `get_class_id` O(1) without scanning the dict on every call. No external dependencies.

**Tech Stack:** Python 3.10+ stdlib only (no numpy, no pandas, no third-party libs).

---

## Constraint

**Only create `src/utils/bbox_utils.py`.** Do not create test files or configuration files.

---

## Key Design Decisions

### `_CLASS_ID_LOOKUP` pre-built reverse map
Rather than scanning `CLASS_MAP` on every `get_class_id` call, build a `dict[(condition, severity) → class_id]` once at module import time. This keeps lookup O(1) and avoids mutating the public constants.

### String normalisation helper `_norm(s)`
All five string-accepting functions need the same normalisation: `.strip().lower().replace(" ", "_")`. Extract as a private one-liner so the logic lives in exactly one place.

### `pixel_to_yolo` clipping scope
Clip the **center coordinates** to [0.0, 1.0] *before* computing bbox corners — an annotation exactly at the image edge (px = 0 or px = img_w-1) can still produce a partially out-of-bounds box if the bbox size is non-zero. Clip the final four values individually.

### `yolo_to_pixel` corner coordinates
`x1 = int((x_center_norm - width_norm / 2) * img_w)` — integer truncation, not rounding. Matches standard YOLO dataset tooling behaviour.

---

## File Structure

```
src/utils/bbox_utils.py     ← only file created
```

---

## Constants reference

```
CLASS_MAP = {
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
```

---

## Task 1: Module skeleton + constants

**Files:**
- Create: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the full file skeleton with all constants**

```python
"""
Bounding-box utilities for the RSNA 2024 Lumbar Spine pipeline.

All bbox sizes are expressed in YOLO-normalised format [0.0, 1.0]
relative to a 640×640 target image.
"""
from __future__ import annotations

BBOX_SIZES: dict[str, tuple[float, float]] = {
    # Sagittal T2 — canal is wide relative to the vertebral body
    "spinal_canal_stenosis":             (0.12, 0.08),
    # Sagittal T1 — foramina are smaller, roughly square
    "left_neural_foraminal_narrowing":   (0.07, 0.07),
    "right_neural_foraminal_narrowing":  (0.07, 0.07),
    # Axial T2 — subarticular zones are small lateral targets
    "left_subarticular_stenosis":        (0.06, 0.06),
    "right_subarticular_stenosis":       (0.06, 0.06),
}

CLASS_MAP: dict[int, tuple[str, str]] = {
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

CLASS_NAMES: list[str] = [
    f"{condition}_{severity}"
    for condition, severity in CLASS_MAP.values()
]

# Pre-built reverse index: (condition, severity) → class_id
_CLASS_ID_LOOKUP: dict[tuple[str, str], int] = {
    v: k for k, v in CLASS_MAP.items()
}


def _norm(s: str) -> str:
    """Lowercase, strip whitespace, replace spaces with underscores."""
    return s.strip().lower().replace(" ", "_")
```

- [ ] **Step 2: Verify import succeeds**

```bash
python3 -c "import src.utils.bbox_utils as b; print(len(b.CLASS_NAMES), 'classes')"
```
Expected: `15 classes`

---

## Task 2: `get_bbox_size`

**Files:**
- Modify: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the function**

```python
def get_bbox_size(condition: str) -> tuple[float, float]:
    """
    Return the normalised (width, height) bbox for a given condition.

    Parameters
    ----------
    condition : str
        Condition name. Case-insensitive; whitespace stripped.

    Returns
    -------
    tuple[float, float]
        ``(width_norm, height_norm)`` in YOLO-normalised [0.0, 1.0] space.

    Raises
    ------
    KeyError
        If ``condition`` is not one of the five recognised conditions.
    """
    key = _norm(condition)
    if key not in BBOX_SIZES:
        raise KeyError(
            f"Unknown condition {condition!r}. "
            f"Valid conditions: {sorted(BBOX_SIZES)}"
        )
    return BBOX_SIZES[key]
```

- [ ] **Step 2: Smoke-test**

```bash
python3 -c "
from src.utils.bbox_utils import get_bbox_size
print(get_bbox_size('spinal_canal_stenosis'))       # (0.12, 0.08)
print(get_bbox_size('  Spinal_Canal_Stenosis  '))   # case+whitespace
try:
    get_bbox_size('bad_condition')
except KeyError as e:
    print('KeyError OK:', e)
"
```
Expected:
```
(0.12, 0.08)
(0.12, 0.08)
KeyError OK: "Unknown condition 'bad_condition'. ..."
```

---

## Task 3: `get_class_id`

**Files:**
- Modify: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the function**

```python
def get_class_id(condition: str, severity: str) -> int:
    """
    Return the YOLO class ID (0–14) for a condition + severity pair.

    Parameters
    ----------
    condition : str
        Condition name. Case-insensitive; whitespace stripped.
    severity : str
        Severity label (``normal_mild``, ``moderate``, or ``severe``).
        Case-insensitive; whitespace stripped.

    Returns
    -------
    int
        Class ID in [0, 14].

    Raises
    ------
    ValueError
        If the (condition, severity) combination is not in CLASS_MAP.
    """
    key = (_norm(condition), _norm(severity))
    if key not in _CLASS_ID_LOOKUP:
        raise ValueError(
            f"Invalid (condition, severity) pair: ({condition!r}, {severity!r}). "
            f"Check CLASS_MAP for valid combinations."
        )
    return _CLASS_ID_LOOKUP[key]
```

- [ ] **Step 2: Smoke-test**

```bash
python3 -c "
from src.utils.bbox_utils import get_class_id
print(get_class_id('spinal_canal_stenosis', 'normal_mild'))        # 0
print(get_class_id('Left_Neural_Foraminal_Narrowing', 'Moderate')) # 7
print(get_class_id('right_subarticular_stenosis', 'severe'))       # 11
try:
    get_class_id('spinal_canal_stenosis', 'critical')
except ValueError as e:
    print('ValueError OK')
"
```
Expected:
```
0
7
11
ValueError OK
```

---

## Task 4: `get_condition_from_class_id`

**Files:**
- Modify: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the function**

```python
def get_condition_from_class_id(class_id: int) -> str:
    """
    Return the condition string for a given YOLO class ID.

    Parameters
    ----------
    class_id : int
        Integer in [0, 14].

    Returns
    -------
    str
        Condition name (e.g. ``"spinal_canal_stenosis"``).

    Raises
    ------
    IndexError
        If ``class_id`` is outside [0, 14].
    """
    if class_id not in CLASS_MAP:
        raise IndexError(
            f"class_id {class_id} is out of range [0, 14]."
        )
    return CLASS_MAP[class_id][0]
```

- [ ] **Step 2: Smoke-test**

```bash
python3 -c "
from src.utils.bbox_utils import get_condition_from_class_id
print(get_condition_from_class_id(0))   # spinal_canal_stenosis
print(get_condition_from_class_id(9))   # right_subarticular_stenosis
print(get_condition_from_class_id(14))  # left_subarticular_stenosis
try:
    get_condition_from_class_id(15)
except IndexError as e:
    print('IndexError OK:', e)
"
```
Expected:
```
spinal_canal_stenosis
right_subarticular_stenosis
left_subarticular_stenosis
IndexError OK: class_id 15 is out of range [0, 14].
```

---

## Task 5: `pixel_to_yolo`

**Files:**
- Modify: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the function**

```python
def pixel_to_yolo(
    x_center_px: float,
    y_center_px: float,
    img_w: int,
    img_h: int,
    condition: str,
) -> tuple[float, float, float, float]:
    """
    Convert a pixel-space center coordinate to a YOLO bounding box.

    Parameters
    ----------
    x_center_px : float
        Horizontal center of the annotation in pixels.
    y_center_px : float
        Vertical center of the annotation in pixels.
    img_w : int
        Image width in pixels (target: 640).
    img_h : int
        Image height in pixels (target: 640).
    condition : str
        Condition name used to look up bbox size via ``get_bbox_size``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(x_center_norm, y_center_norm, width_norm, height_norm)``
        clipped to [0.0, 1.0] and rounded to 6 decimal places.

    Raises
    ------
    KeyError
        Propagated from ``get_bbox_size`` if condition is unrecognised.
    """
    w_norm, h_norm = get_bbox_size(condition)
    x_norm = x_center_px / img_w
    y_norm = y_center_px / img_h
    return (
        round(min(max(x_norm,  0.0), 1.0), 6),
        round(min(max(y_norm,  0.0), 1.0), 6),
        round(min(max(w_norm,  0.0), 1.0), 6),
        round(min(max(h_norm,  0.0), 1.0), 6),
    )
```

- [ ] **Step 2: Smoke-test**

```bash
python3 -c "
from src.utils.bbox_utils import pixel_to_yolo
# Center of 640×640 image, spinal canal stenosis
print(pixel_to_yolo(320, 320, 640, 640, 'spinal_canal_stenosis'))
# Expected: (0.5, 0.5, 0.12, 0.08)

# Out-of-bounds center — should clip to [0,1]
print(pixel_to_yolo(-10, 700, 640, 640, 'left_subarticular_stenosis'))
# Expected: (0.0, 1.0, 0.06, 0.06)
"
```

---

## Task 6: `yolo_to_pixel`

**Files:**
- Modify: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the function**

```python
def yolo_to_pixel(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """
    Convert a YOLO bounding box to pixel-space corner coordinates.

    Parameters
    ----------
    x_center_norm : float
        Normalised horizontal center in [0.0, 1.0].
    y_center_norm : float
        Normalised vertical center in [0.0, 1.0].
    width_norm : float
        Normalised bbox width in [0.0, 1.0].
    height_norm : float
        Normalised bbox height in [0.0, 1.0].
    img_w : int
        Image width in pixels.
    img_h : int
        Image height in pixels.

    Returns
    -------
    tuple[int, int, int, int]
        ``(x1, y1, x2, y2)`` pixel coordinates of bbox corners (top-left,
        bottom-right). Values are integer-truncated, matching standard YOLO
        dataset tooling.

    Raises
    ------
    Nothing — coordinate clamping is the caller's responsibility.
    """
    x1 = int((x_center_norm - width_norm  / 2) * img_w)
    y1 = int((y_center_norm - height_norm / 2) * img_h)
    x2 = int((x_center_norm + width_norm  / 2) * img_w)
    y2 = int((y_center_norm + height_norm / 2) * img_h)
    return x1, y1, x2, y2
```

- [ ] **Step 2: Smoke-test**

```bash
python3 -c "
from src.utils.bbox_utils import yolo_to_pixel
# Center box on 640×640 with spinal canal stenosis size (0.12, 0.08)
print(yolo_to_pixel(0.5, 0.5, 0.12, 0.08, 640, 640))
# Expected: (281, 294, 358, 345)  — (0.5-0.06)*640=281, (0.5-0.04)*640=294
"
```

---

## Task 7: `__main__` self-test block

**Files:**
- Modify: `src/utils/bbox_utils.py`

- [ ] **Step 1: Write the `__main__` block**

```python
if __name__ == "__main__":
    IMG_W, IMG_H = 640, 640
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    errors = 0

    def check(label, got, expected):
        global errors
        ok = got == expected
        print(f"  {'OK' if ok else 'FAIL':4s}  {label}")
        if not ok:
            print(f"        got      {got!r}")
            print(f"        expected {expected!r}")
            errors += 1

    print("=== CLASS_NAMES length ===")
    check("15 class names", len(CLASS_NAMES), 15)

    print("\n=== get_class_id ===")
    check("class 0",  get_class_id("spinal_canal_stenosis", "normal_mild"), 0)
    check("class 2",  get_class_id("spinal_canal_stenosis", "severe"),      2)
    check("class 3",  get_class_id("right_neural_foraminal_narrowing", "normal_mild"), 3)
    check("class 6",  get_class_id("left_neural_foraminal_narrowing",  "normal_mild"), 6)
    check("class 9",  get_class_id("right_subarticular_stenosis", "normal_mild"),      9)
    check("class 12", get_class_id("left_subarticular_stenosis",  "normal_mild"),      12)
    check("class 14", get_class_id("left_subarticular_stenosis",  "severe"),           14)
    check("case-insensitive", get_class_id("Spinal_Canal_Stenosis", "Moderate"),       1)

    print("\n=== get_condition_from_class_id ===")
    check("id 0",  get_condition_from_class_id(0),  "spinal_canal_stenosis")
    check("id 6",  get_condition_from_class_id(6),  "left_neural_foraminal_narrowing")
    check("id 14", get_condition_from_class_id(14), "left_subarticular_stenosis")

    print("\n=== pixel_to_yolo (all 5 conditions) ===")
    center_x, center_y = 320.0, 320.0
    cases = [
        ("spinal_canal_stenosis",            (0.5, 0.5, 0.12, 0.08)),
        ("left_neural_foraminal_narrowing",   (0.5, 0.5, 0.07, 0.07)),
        ("right_neural_foraminal_narrowing",  (0.5, 0.5, 0.07, 0.07)),
        ("left_subarticular_stenosis",        (0.5, 0.5, 0.06, 0.06)),
        ("right_subarticular_stenosis",       (0.5, 0.5, 0.06, 0.06)),
    ]
    for cond, expected in cases:
        result = pixel_to_yolo(center_x, center_y, IMG_W, IMG_H, cond)
        check(cond, result, expected)

    print("\n=== round-trip: pixel → YOLO → pixel (center within ±1 px) ===")
    for cond, _ in cases:
        xc, yc = 200.0, 350.0
        yolo = pixel_to_yolo(xc, yc, IMG_W, IMG_H, cond)
        x1, y1, x2, y2 = yolo_to_pixel(*yolo, IMG_W, IMG_H)
        recovered_x = (x1 + x2) / 2
        recovered_y = (y1 + y2) / 2
        within = abs(recovered_x - xc) <= 1 and abs(recovered_y - yc) <= 1
        label = f"round-trip {cond}"
        ok = within
        print(f"  {'OK' if ok else 'FAIL':4s}  {label}"
              f"  (recovered center: {recovered_x:.1f}, {recovered_y:.1f})")
        if not ok:
            errors += 1

    print("\n=== error handling ===")
    try:
        get_bbox_size("bad_condition")
        print("  FAIL  get_bbox_size should have raised KeyError")
        errors += 1
    except KeyError:
        print("  OK    get_bbox_size raises KeyError for unknown condition")

    try:
        get_class_id("spinal_canal_stenosis", "critical")
        print("  FAIL  get_class_id should have raised ValueError")
        errors += 1
    except ValueError:
        print("  OK    get_class_id raises ValueError for invalid severity")

    try:
        get_condition_from_class_id(15)
        print("  FAIL  get_condition_from_class_id should have raised IndexError")
        errors += 1
    except IndexError:
        print("  OK    get_condition_from_class_id raises IndexError for id 15")

    print(f"\n{'All tests passed.' if errors == 0 else f'{errors} test(s) FAILED.'}")
```

- [ ] **Step 2: Run the self-test**

```bash
python3 -m src.utils.bbox_utils
```
Expected: All lines print `OK`, final line `All tests passed.`

---

## Verification

```bash
# Import check
python3 -c "from src.utils.bbox_utils import BBOX_SIZES, CLASS_MAP, CLASS_NAMES, get_bbox_size, get_class_id, get_condition_from_class_id, pixel_to_yolo, yolo_to_pixel; print('OK')"

# Self-test
python3 -m src.utils.bbox_utils
```

---

## Self-Review Checklist

- [x] `BBOX_SIZES` — 5 conditions, values match spec exactly
- [x] `CLASS_MAP` — 15 entries, ordering matches spec (0-2 spinal, 3-5 right NFN, 6-8 left NFN, 9-11 right sub, 12-14 left sub)
- [x] `CLASS_NAMES` — 15 strings derived from CLASS_MAP, format `"condition_severity"`
- [x] `get_bbox_size` — case-insensitive, KeyError with message
- [x] `get_class_id` — case-insensitive, ValueError with message
- [x] `get_condition_from_class_id` — IndexError if out of range [0, 14]
- [x] `pixel_to_yolo` — divides by img_w/img_h, clips to [0,1], rounds to 6 dp
- [x] `yolo_to_pixel` — returns (x1,y1,x2,y2) as ints via truncation
- [x] `__main__` — covers all 5 conditions, both directions, round-trip ±1px, error paths
- [x] No stdlib-beyond-builtins imports, no src/ imports, no hardcoded 1000×1000
- [x] All functions have type hints and numpy-style docstrings
