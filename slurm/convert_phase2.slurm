#!/bin/bash
#SBATCH --job-name=dcm2niix_phase2
#SBATCH --array=1-45%40
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --partition=batch
#SBATCH --output=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/phase2_%A_%a.out
#SBATCH --error=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/phase2_%A_%a.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0

PROJECT=/home/sak185/dia-endo-conversion
SCRATCH=/scratch/pioneer/users/sak185/dia-endo-conversion

cd "$PROJECT"
source .venv/bin/activate

WORKPLAN=$SCRATCH/workplan/workplan.csv
PATIENT_LIST=$SCRATCH/workplan/subset_phase2.txt
OUT_ROOT=$SCRATCH/output
MANIFEST_PART=$OUT_ROOT/manifest_part_phase2_${SLURM_ARRAY_TASK_ID}.csv

TASK_SCRATCH=${PFSDIR:-/tmp}

# Preflight: fail fast if any source data is missing for the Phase 2 cohort.
python scripts/preflight_check.py \
    --workplan "$WORKPLAN" \
    --patient-list "$PATIENT_LIST" \
    --splits-json "$SCRATCH/workplan/splits.json"

python scripts/convert_one_patient.py \
    --workplan "$WORKPLAN" \
    --patient-list "$PATIENT_LIST" \
    --output-root "$OUT_ROOT" \
    --manifest-part "$MANIFEST_PART" \
    --task-id $SLURM_ARRAY_TASK_ID \
    --task-count 45 \
    --scratch-dir "$TASK_SCRATCH"

echo "Phase 2 task ${SLURM_ARRAY_TASK_ID} done at $(date)"
