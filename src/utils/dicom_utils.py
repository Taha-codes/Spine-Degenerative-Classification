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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_window_value(val) -> float:
    """Extract a scalar float from a DSfloat or MultiValue window tag."""
    if hasattr(val, "__len__") and not isinstance(val, str):
        return float(val[0])
    return float(val)


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 255] uint8 via min/max; returns zeros for flat arrays."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dicom(path: str | Path) -> pydicom.Dataset:
    """
    Load a DICOM file from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the ``.dcm`` file.

    Returns
    -------
    pydicom.Dataset
        Parsed DICOM dataset.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist on disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    return pydicom.dcmread(str(path))


def get_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """
    Extract and rescale the pixel array from a DICOM dataset.

    Applies RescaleSlope and RescaleIntercept if both tags are present.
    The returned array is always ``float32`` regardless of stored bit depth.

    Parameters
    ----------
    ds : pydicom.Dataset
        Loaded DICOM dataset containing pixel data.

    Returns
    -------
    np.ndarray
        2-D float32 array of pixel values after optional rescaling.

    Raises
    ------
    Nothing — missing rescale tags are silently skipped (no rescaling applied).
    """
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        arr = arr * slope + intercept
    return arr


def apply_windowing(
    arr: np.ndarray,
    ds: pydicom.Dataset,
    window_center: float | None = None,
    window_width: float | None = None,
) -> np.ndarray:
    """
    Apply DICOM window/level adjustment, returning a uint8 image.

    Uses explicit ``window_center``/``window_width`` arguments when provided;
    otherwise reads ``WindowCenter``/``WindowWidth`` from the dataset header.
    Falls back to min/max normalization when neither source is available or
    when the effective window width is zero.

    Parameters
    ----------
    arr : np.ndarray
        Input float32 2-D pixel array (from ``get_pixel_array``).
    ds : pydicom.Dataset
        DICOM dataset (used to read window tags when kwargs are None).
    window_center : float or None, optional
        Override for WindowCenter. Uses header tag when None.
    window_width : float or None, optional
        Override for WindowWidth. Uses header tag when None.

    Returns
    -------
    np.ndarray
        uint8 2-D array with values in [0, 255].

    Raises
    ------
    Nothing — logs warnings on missing tags and uses safe fallbacks.
    """
    wc, ww = window_center, window_width

    if wc is None or ww is None:
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = _extract_window_value(ds.WindowCenter)
            ww = _extract_window_value(ds.WindowWidth)
        else:
            logger.warning(
                "WindowCenter/WindowWidth tags absent and no kwargs supplied; "
                "falling back to min/max normalization"
            )
            return _minmax_normalize(arr)

    if ww == 0.0:
        logger.warning(
            "WindowWidth is zero; falling back to min/max normalization"
        )
        return _minmax_normalize(arr)

    lower = wc - ww / 2.0
    upper = wc + ww / 2.0
    clipped = np.clip(arr, lower, upper)
    scaled = (clipped - lower) / (upper - lower) * 255.0
    return scaled.astype(np.uint8)


def apply_clahe(
    arr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast in MRI slices, improving visibility of subtle
    stenosis that would be washed out by global normalization alone.

    Parameters
    ----------
    arr : np.ndarray
        uint8 2-D input image (output of ``apply_windowing``).
    clip_limit : float, optional
        Threshold for contrast limiting. Default is 2.0.
    tile_grid_size : tuple of (int, int), optional
        Size of the grid for histogram equalization. Default is (8, 8).

    Returns
    -------
    np.ndarray
        uint8 2-D array of the same shape as input.

    Raises
    ------
    Nothing — no DICOM tags are involved.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(arr)


def fix_orientation(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Correct image orientation using the ImageOrientationPatient DICOM tag.

    Flips the image left-right if the row-direction x-component is negative,
    and flips up-down if the column-direction y-component is negative. Returns
    the array unchanged when the tag is absent.

    Parameters
    ----------
    arr : np.ndarray
        2-D image array.
    ds : pydicom.Dataset
        Loaded DICOM dataset.

    Returns
    -------
    np.ndarray
        Orientation-corrected 2-D array of the same dtype and shape.

    Raises
    ------
    Nothing — returns unchanged array and logs a warning if the tag is absent.
    """
    if not hasattr(ds, "ImageOrientationPatient"):
        logger.warning(
            "ImageOrientationPatient tag absent; returning array unchanged"
        )
        return arr

    iop = ds.ImageOrientationPatient
    row_x = float(iop[0])  # x-component of row direction cosine
    col_y = float(iop[4])  # y-component of col direction cosine

    if row_x < 0:
        arr = np.fliplr(arr)
    if col_y < 0:
        arr = np.flipud(arr)
    return arr


def get_pixel_spacing(ds: pydicom.Dataset) -> tuple[float, float]:
    """
    Extract pixel spacing from a DICOM dataset.

    Parameters
    ----------
    ds : pydicom.Dataset
        Loaded DICOM dataset.

    Returns
    -------
    tuple[float, float]
        ``(row_spacing, col_spacing)`` in millimetres. Defaults to
        ``(1.0, 1.0)`` if the PixelSpacing tag is absent.

    Raises
    ------
    Nothing — returns safe default and logs a warning on missing tag.
    """
    if not hasattr(ds, "PixelSpacing"):
        logger.warning("PixelSpacing tag absent; defaulting to (1.0, 1.0)")
        return (1.0, 1.0)
    return (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))


def get_series_description(ds: pydicom.Dataset) -> str:
    """
    Extract the series description from a DICOM dataset.

    Used downstream to route images to the correct processing branch
    (sagittal T1, sagittal T2/STIR, or axial T2).

    Parameters
    ----------
    ds : pydicom.Dataset
        Loaded DICOM dataset.

    Returns
    -------
    str
        Stripped, lowercase series description. Returns ``"unknown"`` if the
        SeriesDescription tag is absent.

    Raises
    ------
    Nothing — returns safe default and logs a warning on missing tag.
    """
    if not hasattr(ds, "SeriesDescription"):
        logger.warning("SeriesDescription tag absent; defaulting to 'unknown'")
        return "unknown"
    return str(ds.SeriesDescription).strip().lower()


def dicom_to_numpy(
    path: str | Path,
    apply_window: bool = True,
    apply_clahe_flag: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Load a DICOM file and convert it to a processed numpy array.

    Chains: load_dicom → get_pixel_array → (apply_windowing →
    apply_clahe)? → fix_orientation. When ``apply_window=False``, both
    windowing and CLAHE are skipped (CLAHE requires uint8 input).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the DICOM file.
    apply_window : bool, optional
        If True, apply windowing and CLAHE. If False, skip both.
        Default is True.
    apply_clahe_flag : bool, optional
        If True (and ``apply_window`` is True), apply CLAHE after windowing.
        Default is True.

    Returns
    -------
    tuple[np.ndarray, dict]
        - arr : uint8 2-D array (float32 when ``apply_window=False``).
        - meta : dict with keys ``pixel_spacing``, ``series_description``,
          ``study_instance_uid``, ``series_instance_uid``,
          ``instance_number``, ``rows``, ``cols``.

    Raises
    ------
    FileNotFoundError
        Propagated from ``load_dicom`` when the file does not exist.
    """
    ds = load_dicom(path)
    arr = get_pixel_array(ds)

    if apply_window:
        arr = apply_windowing(arr, ds)
        if apply_clahe_flag:
            arr = apply_clahe(arr)

    arr = fix_orientation(arr, ds)

    def _safe_str(tag_name: str) -> str:
        val = getattr(ds, tag_name, None)
        return str(val).strip() if val is not None else "unknown"

    meta = {
        "pixel_spacing": get_pixel_spacing(ds),
        "series_description": get_series_description(ds),
        "study_instance_uid": _safe_str("StudyInstanceUID"),
        "series_instance_uid": _safe_str("SeriesInstanceUID"),
        "instance_number": _safe_str("InstanceNumber"),
        "rows": getattr(ds, "Rows", arr.shape[0]),
        "cols": getattr(ds, "Columns", arr.shape[1]),
    }
    return arr, meta


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.dicom_utils <path_to_dicom>")
        sys.exit(1)

    _arr, _meta = dicom_to_numpy(sys.argv[1], apply_window=True, apply_clahe_flag=True)
    print(_meta)
    cv2.imwrite("test_output.png", _arr)
    print(f"Saved test_output.png  shape={_arr.shape}  dtype={_arr.dtype}")
