"""Per-slice PNG rendering for visualization (Component 8 §2.4).

Coordinate frame (PRD I.8.8):
    Cached volume: ``(X, Y, Z)`` in RAS. Per-slice (coronal):
    ``volume[:, slice_y, :]`` has shape ``(X_dim, Z_dim)`` with rows=X
    (R-L axis) and cols=Z (I-S axis). Boxes are stored as ``(x1, z1, x2, z2)``.

    The user-requested anatomic coronal display is **rot90 CCW then fliplr
    of the native ``(X, Z)`` slice**, which yields a ``(Z_dim, X_dim)`` image
    with:

        row 0 (top)        ⇒ Z = Z_dim-1   ⇒ SUPERIOR
        row Z_dim-1 (bot.) ⇒ Z = 0          ⇒ INFERIOR
        col 0 (left)       ⇒ X = X_dim-1   ⇒ patient's RIGHT (RAS)
        col X_dim-1 (right)⇒ X = 0          ⇒ patient's LEFT

    The transform applied to the image, mask, and box coordinates in
    lockstep is, for an original pixel at row=x, col=z in ``(X, Z)``:

        new_row = Z_dim - 1 - z     (so z=z1 maps to row Z_dim-1-z1)
        new_col = X_dim - 1 - x     (so x=x1 maps to col X_dim-1-x1)

    For a box ``(x1, z1, x2, z2)``, the transformed AABB is:

        x1' = X_dim - x2            y1' = Z_dim - z2
        x2' = X_dim - x1            y2' = Z_dim - z1

    This delivers a correct radiology-style coronal view (S up, R on
    viewer's left) and keeps boxes/mask overlaid on the lesion.

Colors (per user request):
    - Predictions: red, solid 1.5 px outline.
    - GT boxes: green, dashed 1.5 px outline.
    - Lesion mask overlay: semi-transparent green.

All matplotlib usage is via the ``Agg`` backend (headless). Renders return a
``(H, W, 3) uint8`` RGB array; ``save_slice_png`` writes it to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EventType = Literal["tp", "fp", "fn"]


PRED_COLOR = (1.0, 0.0, 0.0)        # red
GT_COLOR = (0.0, 0.85, 0.0)          # green
MASK_COLOR = (0.0, 0.85, 0.0)        # green
MASK_ALPHA = 0.40


def _extract_slice_image(volume: np.ndarray, slice_y: int) -> np.ndarray:
    """Extract a 2D slice from either a 5-channel stack or a full ``(X,Y,Z)`` volume.

    Returns the slice in its native ``(X, Z)`` orientation. The caller is
    responsible for transposing to ``(Z, X)`` for box-aligned display.
    """
    arr = np.asarray(volume)
    if arr.ndim == 3 and arr.shape[0] in (3, 5, 7) and arr.shape[0] != arr.shape[1]:
        # (C, H, W) channel stack → center channel.
        c = arr.shape[0] // 2
        return arr[c].astype(np.float32)
    if arr.ndim == 3:
        return arr[:, int(slice_y), :].astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported volume shape {arr.shape!r}")


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Percentile-stretch to [0, 1] for display."""
    img = img.astype(np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return out


def _to_xz(arr: np.ndarray, slice_y: int) -> np.ndarray:
    """Reduce a slice/volume/channel-stack to a 2D ``(X, Z)`` array (rows=X,
    cols=Z) — the native cached coronal-slice frame. The anatomic transform
    is applied separately via :func:`_anat_transform_image`.
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (3, 5, 7) and a.shape[0] != a.shape[1]:
        # Channel stack ``(C, Z, X)`` per Component 4 §9 — center channel,
        # then transpose back to ``(X, Z)``.
        a = a[a.shape[0] // 2]
        return a.T.astype(np.float32)
    if a.ndim == 3:
        return a[:, int(slice_y), :].astype(np.float32)
    if a.ndim == 2:
        return a.astype(np.float32)
    raise ValueError(f"Unsupported array shape {a.shape!r}")


def _anat_transform_image(img_xz: np.ndarray) -> np.ndarray:
    """Apply rot90 CCW then flip-horizontally to an ``(X, Z)`` image.

    Output shape is ``(Z_dim, X_dim)`` — radiology-style coronal: S at top,
    patient's R on viewer's left.
    """
    return np.fliplr(np.rot90(img_xz, k=1))


def _anat_transform_box(
    boxes_xz: np.ndarray, x_dim: int, z_dim: int
) -> np.ndarray:
    """Map boxes from native ``(x1, z1, x2, z2)`` coords to the anatomic
    display frame produced by :func:`_anat_transform_image`.

    For an original pixel at row=x, col=z in ``(X, Z)``:
        new_row = Z_dim - 1 - z
        new_col = X_dim - 1 - x

    For a box AABB:
        x1' = X_dim - x2     y1' = Z_dim - z2
        x2' = X_dim - x1     y2' = Z_dim - z1
    """
    if boxes_xz.size == 0:
        return boxes_xz.reshape(-1, 4).astype(np.float32, copy=False)
    b = np.asarray(boxes_xz, dtype=np.float32).reshape(-1, 4)
    x1, z1, x2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    new_x1 = x_dim - x2
    new_y1 = z_dim - z2
    new_x2 = x_dim - x1
    new_y2 = z_dim - z1
    return np.stack([new_x1, new_y1, new_x2, new_y2], axis=1).astype(np.float32)


def render_slice_overlay(
    volume: np.ndarray,
    slice_y: int,
    lesion_mask_center: np.ndarray | None,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    event_type: EventType,
    patient_id: str | None = None,
    fig_size_px: int = 512,
    apply_anat_orientation: bool = True,
) -> np.ndarray:
    """Render a single slice with prediction + GT box overlays.

    Returns ``(H, W, 3)`` uint8 RGB image.
    """
    img_xz = _to_xz(volume, slice_y)
    img_norm = _normalize_for_display(img_xz)
    X_dim, Z_dim = img_norm.shape

    # Mask in (X, Z) frame, aligned to the image.
    mask_xz: np.ndarray | None = None
    if lesion_mask_center is not None:
        mask_xz = _to_xz(lesion_mask_center, slice_y)
        if mask_xz.shape != img_norm.shape:
            mask_xz = None  # silent skip on shape mismatch

    pred_boxes_arr = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores_arr = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes_arr = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    if apply_anat_orientation:
        img_disp = _anat_transform_image(img_norm)
        if mask_xz is not None:
            mask_disp = _anat_transform_image(mask_xz)
        else:
            mask_disp = None
        pred_boxes_disp = _anat_transform_box(pred_boxes_arr, X_dim, Z_dim)
        gt_boxes_disp = _anat_transform_box(gt_boxes_arr, X_dim, Z_dim)
    else:
        # Non-anatomic path: still need rows=Z, cols=X so boxes overlay mask.
        img_disp = img_norm.T
        mask_disp = mask_xz.T if mask_xz is not None else None
        pred_boxes_disp = pred_boxes_arr
        gt_boxes_disp = gt_boxes_arr

    H, W = img_disp.shape

    dpi = 100
    fig_in = fig_size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.imshow(img_disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    # Lesion mask overlay (semi-transparent green).
    if mask_disp is not None:
        mask_bool = (mask_disp > 0.5).astype(np.float32)
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., 0] = MASK_COLOR[0]
        rgba[..., 1] = MASK_COLOR[1]
        rgba[..., 2] = MASK_COLOR[2]
        rgba[..., 3] = mask_bool * MASK_ALPHA
        ax.imshow(rgba, interpolation="nearest")

    # Predicted boxes (red, solid).
    for box, score in zip(pred_boxes_disp, pred_scores_arr):
        x1, y1, x2, y2 = box.tolist()
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=PRED_COLOR, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(0.0, y1 - 2),
            f"{float(score):.2f}",
            color=PRED_COLOR, fontsize=6,
            verticalalignment="bottom",
        )

    # GT boxes (green, dashed).
    for box in gt_boxes_disp:
        x1, y1, x2, y2 = box.tolist()
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=GT_COLOR,
            linestyle="--", facecolor="none",
        )
        ax.add_patch(rect)

    title_pid = f"{patient_id} | " if patient_id else ""
    ax.set_title(
        f"{title_pid}slice y={int(slice_y)} | {event_type.upper()}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def save_slice_png(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an RGB ``(H, W, 3)`` uint8 image to ``output_path`` as PNG."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    import matplotlib.image as mpimg

    mpimg.imsave(str(out), arr)
    return out
