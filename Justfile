SCRATCH := "/scratch/pioneer/users/sak185/dia-endo-conversion"

# List available recipes
default:
    @just --list

# Monitor a running SLURM array conversion job by name.
# Resolves the job ID from squeue, then streams progress via scripts/monitor.py.
# Usage:
#   just monitor dcm2niix_phase1
#   just monitor dcm2niix_phase2
monitor JOB_NAME:
    #!/usr/bin/env bash
    set -euo pipefail
    JOBID=$(squeue -u "$USER" --name={{JOB_NAME}} --noheader -o "%A" 2>/dev/null | sort -u | head -1)
    if [ -z "$JOBID" ]; then
        echo "ERROR: no running job found with name '{{JOB_NAME}}' for user $USER" >&2
        echo "  Currently in queue:" >&2
        squeue -u "$USER" -o "  %.10i %.20j %.8T %.10M" >&2
        exit 1
    fi
    case "{{JOB_NAME}}" in
        dcm2niix_phase1)
            LIST_ARG="--patient-list {{SCRATCH}}/workplan/subset_phase1.txt"
            GLOB_ARG="--manifest-glob manifest_part_phase1_*.csv" ;;
        dcm2niix_phase2)
            LIST_ARG="--patient-list {{SCRATCH}}/workplan/subset_phase2.txt"
            GLOB_ARG="--manifest-glob manifest_part_phase2_*.csv" ;;
        *)
            LIST_ARG=""
            GLOB_ARG="" ;;
    esac
    echo "Resolved job '{{JOB_NAME}}' -> JOBID=$JOBID"
    uv run python scripts/monitor.py \
        --job-id "$JOBID" \
        --workplan {{SCRATCH}}/workplan/workplan.csv \
        --output-root {{SCRATCH}}/output \
        --cohort all \
        $LIST_ARG $GLOB_ARG
