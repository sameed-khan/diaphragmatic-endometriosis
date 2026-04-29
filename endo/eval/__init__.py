"""Component 7 — Post-training evaluation.

Public APIs:

- ``weighted_box_fusion_3d`` (``endo.eval.wbf``)
- ``compute_froc`` (``endo.eval.froc``)
- ``compute_volume_metrics``, ``bootstrap_ci`` (``endo.eval.metrics``)
- ``grid_search_threshold`` (``endo.eval.threshold_search``)
- ``stratify_metrics`` (``endo.eval.stratified``)
- ``append_eval_report``, ``write_eval_thresholds_json`` (``endo.eval.report``)
- ``run_cv_evaluation``, ``run_holdout_inference`` (``endo.eval.run_eval``)
"""

from endo.eval.froc import compute_froc
from endo.eval.metrics import bootstrap_ci, compute_volume_metrics
from endo.eval.report import (
    EvalReportRow,
    append_eval_report,
    write_eval_thresholds_json,
)
from endo.eval.run_eval import run_cv_evaluation, run_holdout_inference
from endo.eval.stratified import stratify_metrics
from endo.eval.threshold_search import grid_search_threshold
from endo.eval.wbf import weighted_box_fusion_3d

__all__ = [
    "EvalReportRow",
    "append_eval_report",
    "bootstrap_ci",
    "compute_froc",
    "compute_volume_metrics",
    "grid_search_threshold",
    "run_cv_evaluation",
    "run_holdout_inference",
    "stratify_metrics",
    "weighted_box_fusion_3d",
    "write_eval_thresholds_json",
]
