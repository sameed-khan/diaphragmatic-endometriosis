"""Constants and helpers shared across scripts."""
from pathlib import Path

# Patients to exclude from negatives (also appear in positives in /home/jjs374/DiaE).
# Kept here for reference + dedup at consolidation time. By the time prescan runs,
# the consolidated /scratch/.../input/{positive,negative}/ tree already excludes
# these from the negative cohort.
POSITIVE_OVERLAP_IDS = {
    "ANON4EF24D0EDFA5",
    "ANON5DCB62C77550",
    "ANON7317255BC6B3",
    "ANONA9B87788C42B",
}

# Patients excluded from the study entirely (neither positive nor negative).
# These are positives whose only existing mask was made on a non-canonical
# WATER series, AND on visual inspection the lesion is not visible on the
# canonical series — so a manual canonical re-mask is not feasible. They are
# dropped at workplan build time and logged to skipped.csv with reason
# "excluded_no_visible_lesion_on_canonical". Re-introduce by removing from
# this set once the underlying issue is resolved.
EXCLUDED_PIDS = {
    "ANON25C6C345BBDA",
    "ANON474B6A632EC1",
    "ANONB37185FC9DAF",
    "ANONC0DC7E3FB015",
    "ANONC4A3AEBA378D",
}

# New canonical layout (post-consolidation):
#   <input_root>/positive/<ANONID>/<series>/...
#   <input_root>/negative/<ANONID>/<series>/...
#   <input_root>/nifti/<ANONID>[_<series>].nii.gz
#   <input_root>/masks/<ANONID>[_<series>].{nii.gz,csv}
COHORT_DIRS = {"positive": "pos", "negative": "neg"}

IMAGETYPE_TO_ROLE = {
    "WATER":     "water",
    "FAT":       "fat",
    "IN_PHASE":  "inphase",
    "OUT_PHASE": "outphase",
}

MIN_SLICES_QC_FLAG = 30


def first_dcm_in(series_dir: Path) -> Path | None:
    """Return the first .dcm-like file in a series folder, or None if empty.
    Skips dotfiles (e.g. .DS_Store)."""
    for p in sorted(series_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() == ".dcm" or "." not in p.name:
            return p
    return None
