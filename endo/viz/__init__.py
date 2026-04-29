"""Per-slice prediction visualization (Component 8 §2)."""

from endo.viz.render import render_slice_overlay, save_slice_png
from endo.viz.tagging import tag_slice_events

__all__ = [
    "render_slice_overlay",
    "save_slice_png",
    "tag_slice_events",
]
