"""Component 4 — training augmentation stack.

Exports the public ``TrainAugmentation`` callable plus the per-stage helpers
used by the unit tests.
"""

from __future__ import annotations

from endo.augmentation.boxes import (
    clamp_box_to_frame,
    derive_all_boxes,
    derive_boxes_from_mask,
    read_connectivity,
)
from endo.augmentation.geometric import (
    apply_affine_lockstep,
    apply_elastic_lockstep,
    geometric_aug,
    random_affine_2d,
    random_elastic_2d,
)
from endo.augmentation.intensity import (
    intensity_aug,
    random_brightness_contrast,
    random_gamma,
    random_gaussian_noise,
)
from endo.augmentation.paste import (
    apply_paste,
    multi_paste_volume,
    sample_n_pastes,
    select_paste_site,
)
from endo.augmentation.transform import (
    TrainAugmentation,
    compute_cohort_local_std,
)


__all__ = [
    "TrainAugmentation",
    "apply_affine_lockstep",
    "apply_elastic_lockstep",
    "apply_paste",
    "clamp_box_to_frame",
    "compute_cohort_local_std",
    "derive_all_boxes",
    "derive_boxes_from_mask",
    "geometric_aug",
    "intensity_aug",
    "multi_paste_volume",
    "random_affine_2d",
    "random_brightness_contrast",
    "random_elastic_2d",
    "random_gamma",
    "random_gaussian_noise",
    "read_connectivity",
    "sample_n_pastes",
    "select_paste_site",
]
