# GPU Starvation Profiling — Notes & Interpretation

This project includes `scripts/profile_gpu_starvation.py` to quantify CPU vs GPU time and identify pipeline stalls. The script produces structured JSON (stdout and optional file) with per‑step timing and GPU utilization summaries.

## Why this is “scientifically precise”

- Uses `torch.cuda.synchronize()` to measure GPU timing boundaries (no async skew).
- Reports distributions (p50/p90/p99) rather than single averages.
- Uses NVML (`nvidia-ml-py`) for GPU utilization and memory sampling during the run.
- Separately times dataloader batch production and GPU forward/backward/optimizer steps.

## Key Outputs

- `cpu_batch_time_s`: time to pull a batch from the dataloader.
- `h2d_time_s`: host‑to‑device transfer time.
- `gpu_step_time_s`: forward + backward + optimizer step (synchronized).
- `step_time_s`: end‑to‑end step wall time.
- `gpu_util`: NVML GPU utilization (percent).
- `gpu_mem_used_mb`: NVML memory usage snapshot (MB).

## How to run

```bash
# Dataloader‑only throughput (no GPU)
uv run python scripts/profile_gpu_starvation.py \
  --experiment experiments/smoke.py --fold 0 --mode dataloader \
  --num-batches 200 --workers 8 --pin-memory

# End‑to‑end training step profiling
uv run python scripts/profile_gpu_starvation.py \
  --experiment experiments/smoke.py --fold 0 --mode training \
  --steps 200 --warmup 20 --workers 8 --pin-memory
```

## How to interpret

- **GPU starvation** is likely if `cpu_batch_time_s` is similar to or larger than `gpu_step_time_s`, and GPU util p50/p90 is low (<50%).
- **H2D bound** if `h2d_time_s` is a large fraction of `step_time_s` (suggests pin_memory or batch size issues).
- **Compute bound** if `gpu_step_time_s` dominates and GPU util is high.

## Common fixes to try (minimal changes)

1. Increase dataloader `num_workers` and `prefetch_factor`.
2. Reduce heavy CPU augmentation (elastic/paste frequency).
3. Avoid double copies in augmentation (single float32 conversion).
4. Update the EMA score tracker less frequently to reduce per‑step NMS overhead.
5. Ensure `pin_memory=True` and `non_blocking=True` on `.to(device)` calls.

## Optional advanced profiling (NVIDIA tools)

For deeper GPU pipeline insight, NVIDIA’s Nsight Systems can be used externally:

```bash
nsys profile -t cuda,nvtx,osrt -o nsys_profile \
  uv run python scripts/profile_gpu_starvation.py \
  --experiment experiments/smoke.py --fold 0 --mode training --steps 100
```

This captures kernel timelines, CPU‑GPU overlap, and dataloader stalls. Use this only when needed; it adds overhead.
