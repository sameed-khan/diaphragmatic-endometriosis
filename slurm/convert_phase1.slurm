#!/bin/bash
#SBATCH --job-name=dcm2niix_phase1
#SBATCH --array=1-10%20
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --output=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/phase1_%A_%a.out
#SBATCH --error=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/phase1_%A_%a.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0

PROJECT=/home/sak185/dia-endo-conversion
SCRATCH=/scratch/pioneer/users/sak185/dia-endo-conversion

cd "$PROJECT"
source .venv/bin/activate

WORKPLAN=$SCRATCH/workplan/workplan.csv
PATIENT_LIST=$SCRATCH/workplan/subset_phase1.txt
OUT_ROOT=$SCRATCH/output
MANIFEST_PART=$OUT_ROOT/manifest_part_phase1_${SLURM_ARRAY_TASK_ID}.csv

TASK_SCRATCH=${PFSDIR:-/tmp}

# Preflight: fail fast if any source data is missing for the Phase 1 cohort.
# Cheap (single Python invocation, no dcm2niix); aborts the array task with
# a clear error before we waste compute on FileNotFoundError loops.
python scripts/preflight_check.py \
    --workplan "$WORKPLAN" \
    --patient-list "$PATIENT_LIST" \
    --splits-json "$SCRATCH/workplan/splits.json"

# convert_one_patient.py applies the patient list filter first, then partitions
# the remaining rows round-robin across array tasks (task_count = array size).
python scripts/convert_one_patient.py \
    --workplan "$WORKPLAN" \
    --patient-list "$PATIENT_LIST" \
    --output-root "$OUT_ROOT" \
    --manifest-part "$MANIFEST_PART" \
    --task-id $SLURM_ARRAY_TASK_ID \
    --task-count 10 \
    --scratch-dir "$TASK_SCRATCH"

echo "Phase 1 task ${SLURM_ARRAY_TASK_ID} done at $(date)"
