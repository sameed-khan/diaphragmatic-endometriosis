# Peak VRAM estimate — production training

**TL;DR (conservative):** The full `baseline_rtmdet_p2` training step needs **≈ 6 GiB peak VRAM** at the configured `batch_size=8`, `precision=bf16-mixed`. With a 50 % safety buffer, **size each parallel fold at ≥ 9 GiB**. **A100-80GB MIG with five `1g.10gb` slices (10 GiB each) is comfortably feasible.** A100-**40GB** MIG (`1g.5gb` = 5 GiB per slice) is **not** — peak training already exceeds the slice size.

## Method

Direct measurement, not analytic estimate. `scripts/probe_peak_vram.py` instantiates `LesionDetector` from `experiments/baseline_rtmdet_p2.py`, allocates the `AdamW` optimizer + `ModelEmaV3` shadow (matching `endo/ema_callback.py`), runs synthetic forward + backward + `optim.step()` + `ema.update()` under `torch.autocast(bf16)` for 3 steps, then runs a validation forward with `model.eval()`/`no_grad`. We read `torch.cuda.max_memory_reserved()` because that is the true bound the allocator pins.

Probe ran on the only available device, an A10 (24 GiB).

## Measurements

| phase | bs | precision | allocated peak (MiB) | **reserved peak (MiB)** | notes |
|---|---|---|---:|---:|---|
| model + optimizer + EMA loaded | 8 | bf16-mixed | 309 | 338 | 40.3 M params · fp32 weights · fp32 EMA · no Adam state yet |
| **train step (production)** | **8** | **bf16-mixed** | **4 855** | **5 074** | ≈ 4.95 GiB — peak across 3 steps |
| validation forward | 8 | bf16-mixed | 1 868 | 2 914 | model.eval(), no_grad |
| train step (stress) | 16 | bf16-mixed | 8 939 | 9 658 | ≈ 9.43 GiB |
| validation forward | 16 | bf16-mixed | 2 826 | 4 426 | inference_batch_size=16 path |

Raw probe outputs: `agent/vram_probe.json`, `agent/vram_probe_bs16.json`.

## What is included in the train-step peak

- 40.3 M trainable params (fp32 master weights) — 154 MiB
- AdamW first/second moments (fp32, fp32) — ≈ 308 MiB
- Gradients (fp32) — ≈ 154 MiB
- EMA shadow (fp32, no grad, ema_decay=0.999) — 154 MiB
- Activations (bf16) at B=8, 5×384×384 input through ConvNeXt-tiny stages 0–3 (strides 4/8/16/32) + 4-level FPN (256 ch) + RTMDet head (2 stacked convs, 256 ch, 4 levels) + aux seg head upsampled to 384² — ≈ 4.0 GiB at peak step
- cudnn workspace + allocator fragmentation reserve — included in `reserved` but not `allocated`

The `reserved − allocated` gap (~220 MiB at bs=8) is small; this model is not allocator-fragmentation-bound.

## Conservative ceiling for sizing decisions

Take the measured `reserved` peak and add headroom for:

1. **5 % allocator drift** across long runs (compounds with pos/neg frac shifts changing batch composition).
2. **Augmentation-driven temporary tensors** during copy-paste (`endo/augmentation/`) — these run on CPU per the data pipeline, but the safer assumption is +200 MiB GPU-side scratch.
3. **Periodic deep-eval pass** (`endo/sampler/periodic_eval.py`) at `eval.inference_batch_size=16`. Deep eval runs after a training epoch completes, so training activations are already freed; peak during this phase is bounded by `weights + EMA + Adam state + val(bs=16) forward` ≈ 4.4 GiB measured. **Lower than the training peak** — not the constraint.
4. **Validation between training epochs** at the training `batch_size` — also measured lower than the train peak.
5. **GRU stage** (`endo/gru/`) — runs on FPN-stage features (768-D vectors); negligible compared to the detector.

→ **Conservative train-step peak: 6 GiB at bs=8 / bf16-mixed.** (5.07 measured + ≈ 0.9 GiB margin.)

## A100 + MIG feasibility for parallel 5-fold

Lambda Labs single-GPU A100 instances come in two flavors:

| machine | GPU | MIG profile for 5 parallel folds | per-slice mem | per-slice SMs | feasible? |
|---|---|---|---|---|---|
| `gpu_1x_a100` | A100 40GB | `1g.5gb` × 5 | **≈ 4.75 GiB** usable | ~14 | **NO** — train peak (≈ 6 GiB) exceeds slice |
| `gpu_1x_a100` | A100 40GB | `2g.10gb` × 3 | ≈ 9.5 GiB | ~28 | partial — only 3 folds in parallel |
| `gpu_1x_a100_sxm4` | A100 80GB SXM | `1g.10gb` × 5 (of 7) | ≈ 9.75 GiB usable | ~14 | **YES** — fits with 50 % margin |
| `gpu_1x_a100_sxm4` | A100 80GB SXM | `2g.20gb` × 3 | ≈ 19.5 GiB | ~28 | overkill, only 3 in parallel |

(Per-slice usable memory is slightly less than the nominal — MIG reserves a small overhead. Numbers above use NVIDIA's published usable values.)

### Throughput caveats (not VRAM, but relevant to the decision)

- A `1g.10gb` slice has ~14 SMs vs. 108 on a full A100. Each fold will run at roughly **1/7 the SM throughput of an unpartitioned A100**. Five folds in parallel ≈ **5/7 of full-A100 throughput aggregate**, vs. running folds sequentially on the full GPU at 7/7. So the *only* speedup you get from MIG is in steps where the workload is small enough that one fold can't saturate a full A100 — which is unlikely here given bs=8 + ConvNeXt-tiny.
- **Suggested alternative:** time-share without MIG (5 folds sequentially on full A100, each ≈ 1.4× faster than one fold on MIG slice) — likely ≈ same wall-clock as 5×MIG, simpler to operate, no driver/SLURM/MIG-config overhead. The principal MIG win is *resource isolation* (one fold can't OOM another, no CUDA-context contention), not speed.
- If you do go MIG: each fold needs its own `CUDA_VISIBLE_DEVICES=MIG-<UUID>` env var; the trainer code today assumes a single device, so this should "just work" without code changes — confirm by spot-checking the Lightning `accelerator/devices` setup is `"gpu"/1` (it is, in `endo/cli/run_experiment.py`).

### Headroom you have on `1g.10gb`

| risk | budget impact |
|---|---|
| measured train peak (bs=8, bf16) | 5.1 GiB |
| +20 % cudnn workspace drift | +1.0 GiB |
| +deep-eval inference batch surge | already ≤ train peak, no add |
| +EMA swap during val (already counted) | 0 |
| **conservative used** | **≈ 6.1 GiB** |
| **slice capacity** | **9.75 GiB** |
| **margin** | **~37 %** |

## Honest disclaimers

- The probe ran on **A10** (different SM count and L2 cache vs A100). VRAM consumption is essentially independent of SM count; it depends on tensor shapes and dtype, which match. So the **measurement transfers to A100 directly**.
- Probe used **synthetic batches** with one box per image. Real `batch.boxes` lists vary in length and the RTMDet head's loss assigner allocates target tensors proportional to total box count. With paste augmentation (`n_paste_max=7`), peak boxes per image can hit ~8. The assigner's K×N similarity matrices grow with box count; per-image cost is small but I rounded up the conservative ceiling to absorb it.
- I did **not** stress every callback (paste lesion bank, hard pool refresh, deep eval). The implicated tensors are CPU-resident or run as a separate eval pass that comes in lower than train peak.
- Ground truth: spot-check `nvidia-smi -i 0 --query-gpu=memory.used --format=csv -l 5` during the first epoch of the real run — if it diverges materially from this estimate, replan MIG slicing.

## Recommendation

- **Go A100-80GB SXM with 5×`1g.10gb`** for parallel 5-fold. Peak VRAM safely fits.
- **Do not pick A100-40GB** for this layout — `1g.5gb` is too tight. (40GB only works at 3 parallel folds on `2g.10gb`, which defeats the purpose.)
- Reconsider whether MIG actually wins on wall-clock vs. sequential full-A100 folds; the SM split likely makes them comparable. The strongest case for MIG is process isolation across folds.
