# Component 6.5 — GRU Rescorer (Stage 2)

**Status:** Spec locked, ready for implementation.
**Owner files:** `src/gru_feature_cache.py`, `src/gru_rescorer.py`, `train_gru.py`
**Date:** 2026-04-27
**Companion:** Implements §11 of `agent/training_pipeline_decisions_phase1.md`. Runs sequentially after all 5 detector folds (Component 6) finish. Outputs consumed by Component 7's final eval (with `--use-gru` flag).

---

## 1. Purpose

Train a small bidirectional GRU on **frozen-detector features** to lift volume-level AUROC by exploiting through-plane sequential context that the per-slice detector ignores. The GRU is supervised only on volume-level binary labels (no per-slice presence labels needed). At inference time, GRU outputs `p_t` per slice; each detector box's score is multiplied by `p_t` from its slice; final volume score = `max` over post-WBF rescored box confidences.

---

## 2. Scope

**In scope:**

- One-time feature extraction per fold: run frozen detector on every train+val volume, GAP-pool the last backbone stage feature per slice, cache to disk.
- GRU model, training loop, and per-fold checkpoint.
- Score multiplication helper used by Component 7.
- Light test plan (Stage 2 is small).

**Out of scope:**

- Holdout feature extraction — that's part of Component 7's holdout inference script.
- Volume metrics computation, FROC, AUROC — Component 7.

---

## 3. Inputs

| Input | Path | Used for |
|---|---|---|
| Fold-f detector checkpoint | `runs/baseline_fold{f}/ckpts/<best_epoch>.ckpt` | Frozen feature extractor |
| Cached volumes | `cache/v1/volumes/<patient_id>/volume.npy` | Run inference |
| Splits | `data/splits.json` | Determine fold-f train + val patient lists |
| Manifest | `cache/v1/preprocessed_manifest.csv` | Patient labels, cohort, fold |

---

## 4. Outputs (downstream contract)

```
cache/v1/gru_features/
└── fold{0..4}/
    └── <patient_id>.npz
        # Arrays:
        #   feats: (N_valid_slices, 768) float16  — GAP-pooled stage-3 backbone features
        #   slice_ys: (N_valid_slices,) int32     — y indices in cropped (384, 160, 384) frame
        #   patient_label: () int8                 — 0 (negative) or 1 (positive volume)

cache/v1/gru_ckpts/
├── fold{0..4}.pt          # GRU state dict + arch config
└── gru_provenance.json    # build metadata, train metrics per fold
```

Per-fold disk: ~120 MB (486 volumes × 150 slices × 768 × 2 B). Total across 5 folds: ~600 MB.

---

## 5. Feature extraction (Phase 1)

```python
# src/gru_feature_cache.py

@dataclass(frozen=True)
class FeatureCacheConfig:
    fold: int
    detector_ckpt_path: Path
    cache_root: Path
    output_dir: Path                     # cache/v1/gru_features/fold{fold}/
    batch_size: int = 16
    num_workers: int = 4

def extract_features_for_fold(cfg: FeatureCacheConfig):
    """Run frozen detector backbone over every train + val patient in fold;
       GAP-pool the LAST backbone stage; write per-patient .npz."""

    # 1. Load detector with EMA weights
    lm = LesionDetectorLM.load_from_checkpoint(cfg.detector_ckpt_path, strict=False)
    if "ema_state_dict" in torch.load(cfg.detector_ckpt_path):
        lm.model.load_state_dict(torch.load(cfg.detector_ckpt_path)["ema_state_dict"])
    lm.eval().cuda()

    # 2. Build inference DataModule (no aug, allow_holdout=False)
    dm = LesionDataModule(
        cache_root=cfg.cache_root,
        splits_path=Path("data/splits.json"),
        fold=cfg.fold,
        allow_holdout=False,
    )
    dm.setup(stage="fit")

    train_pids = sorted(set(dm.train_patient_ids))
    val_pids = sorted(set(dm.val_patient_ids))
    all_pids = train_pids + val_pids

    # 3. For each patient, iterate slices and extract GAP'd backbone features
    for pid in all_pids:
        loader = dm.inference_dataloader([pid])   # yields one slice at a time
        feats_per_slice = []
        slice_ys = []

        with torch.no_grad():
            for batch in loader:
                x = batch.volume_5ch.cuda()              # (B, 5, 384, 384)
                backbone_feats = lm.model.backbone(x)    # list of 4 stage outputs
                stage3 = backbone_feats[-1]              # (B, 768, 12, 12) — LAST stage
                gap = stage3.mean(dim=(2, 3))            # (B, 768)
                feats_per_slice.append(gap.cpu().numpy().astype(np.float16))
                slice_ys.extend(batch.slice_ys.tolist())

        feats = np.concatenate(feats_per_slice, axis=0)   # (N, 768) fp16
        patient_label = int(dm.manifest_lookup[pid]["label"] == "positive")

        np.savez_compressed(
            cfg.output_dir / f"{pid}.npz",
            feats=feats,
            slice_ys=np.array(slice_ys, dtype=np.int32),
            patient_label=np.int8(patient_label),
        )
```

**Backbone-only inference, no FPN, no head, no aux seg.** Significantly faster than full forward — frozen-detector inference here is ~50% the cost of training-loop forward+backward.

---

## 6. GRU model

```python
# src/gru_rescorer.py

@dataclass(frozen=True)
class GRUConfig:
    input_dim: int = 768
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout_input: float = 0.3
    output_dim: int = 1   # binary presence per slice

class GRURescorer(nn.Module):
    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.input_dropout = nn.Dropout(cfg.dropout_input)
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        gru_out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.head = nn.Linear(gru_out_dim, cfg.output_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, N_slices, 768). Returns (B, N_slices) per-slice presence logits."""
        x = self.input_dropout(feats)
        h, _ = self.gru(x)                # (B, N, 256)
        logits = self.head(h).squeeze(-1) # (B, N)
        return logits

    @torch.no_grad()
    def per_slice_probabilities(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(feats))
```

---

## 7. Training (Phase 2)

```python
@dataclass(frozen=True)
class GRUTrainConfig:
    fold: int
    feature_cache_dir: Path        # cache/v1/gru_features/fold{fold}/
    output_ckpt_path: Path
    splits_path: Path = Path("data/splits.json")
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 16           # patient-level batches
    top_k_for_aux_loss: int = 5
    aux_loss_weight: float = 0.1
    seed: int = 42

class GRUDataset(Dataset):
    """Per-patient feature loader."""
    def __init__(self, patient_ids: list[str], feature_cache_dir: Path):
        self.entries = []
        for pid in patient_ids:
            data = np.load(feature_cache_dir / f"{pid}.npz")
            self.entries.append((pid, data["feats"], int(data["patient_label"])))

    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        pid, feats, label = self.entries[idx]
        return {
            "patient_id": pid,
            "feats": torch.from_numpy(feats).float(),   # (N, 768)
            "label": torch.tensor(label, dtype=torch.float32),
        }

def gru_collate(batch):
    """Pad variable-length slice sequences; return mask."""
    feats = [b["feats"] for b in batch]
    lengths = torch.tensor([f.shape[0] for f in feats])
    feats_padded = pad_sequence(feats, batch_first=True)   # (B, max_N, 768)
    mask = torch.arange(feats_padded.shape[1])[None, :] < lengths[:, None]   # (B, max_N) bool
    labels = torch.stack([b["label"] for b in batch])
    return {"feats": feats_padded, "mask": mask, "labels": labels, "lengths": lengths}

def volume_score(per_slice_probs: torch.Tensor, mask: torch.Tensor, top_k: int = 5):
    """Returns (B,) volume scores via masked max and masked top-k mean."""
    masked = per_slice_probs.masked_fill(~mask, -1.0)
    vol_max = masked.max(dim=1).values
    # top-k mean (ignore padding)
    topk_vals, _ = masked.topk(min(top_k, masked.shape[1]), dim=1)
    vol_topk_mean = topk_vals.mean(dim=1)
    return vol_max, vol_topk_mean

def train_gru_for_fold(cfg: GRUTrainConfig):
    pl.seed_everything(cfg.seed)

    # Load splits, build train/val patient lists
    splits = json.loads(cfg.splits_path.read_text())
    train_pids = [p for p in splits["folds"] if splits["folds"][p] != cfg.fold and splits["cohort"][p] == "cross-validation"]
    val_pids   = [p for p in splits["folds"] if splits["folds"][p] == cfg.fold and splits["cohort"][p] == "cross-validation"]

    train_ds = GRUDataset(train_pids, cfg.feature_cache_dir)
    val_ds   = GRUDataset(val_pids,   cfg.feature_cache_dir)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=gru_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=gru_collate, num_workers=2)

    model = GRURescorer(GRUConfig()).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val_auroc = 0.0
    for epoch in range(cfg.epochs):
        # Train
        model.train()
        for batch in train_loader:
            feats, mask, labels = batch["feats"].cuda(), batch["mask"].cuda(), batch["labels"].cuda()
            logits = model(feats)
            probs = torch.sigmoid(logits)
            vol_max, vol_topk_mean = volume_score(probs, mask, cfg.top_k_for_aux_loss)
            loss_max = bce(torch.logit(vol_max.clamp(1e-6, 1-1e-6)), labels)
            loss_topk = bce(torch.logit(vol_topk_mean.clamp(1e-6, 1-1e-6)), labels)
            loss = loss_max + cfg.aux_loss_weight * loss_topk
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Val
        model.eval()
        scores, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats, mask, labels = batch["feats"].cuda(), batch["mask"].cuda(), batch["labels"].cuda()
                logits = model(feats)
                probs = torch.sigmoid(logits)
                vol_max, _ = volume_score(probs, mask, cfg.top_k_for_aux_loss)
                scores.extend(vol_max.cpu().numpy().tolist())
                gts.extend(labels.cpu().numpy().tolist())
        val_auroc = roc_auc_score(gts, scores)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                "state_dict": model.state_dict(),
                "config": dataclasses.asdict(GRUConfig()),
                "epoch": epoch,
                "val_auroc": val_auroc,
            }, cfg.output_ckpt_path)

    return {"best_val_auroc": best_val_auroc}
```

---

## 8. Rescoring helper (consumed by Component 7)

```python
# src/gru_rescorer.py

def rescore_detector_outputs(
    gru_ckpt_path: Path,
    feature_cache_path: Path,        # one patient's .npz
    detector_boxes_per_slice: dict[int, dict],  # {slice_y: {boxes, scores}}
) -> dict[int, dict]:
    """For each detector box on slice_y_t, multiply its score by p_t from the GRU.
       Returns same-shaped dict with scores replaced by s' = s * p_t."""

    ckpt = torch.load(gru_ckpt_path)
    model = GRURescorer(GRUConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feats = torch.from_numpy(np.load(feature_cache_path)["feats"]).float().unsqueeze(0)  # (1, N, 768)
    slice_ys = np.load(feature_cache_path)["slice_ys"]
    with torch.no_grad():
        per_slice_p = model.per_slice_probabilities(feats).squeeze(0).numpy()    # (N,)

    p_by_slice = dict(zip(slice_ys.tolist(), per_slice_p.tolist()))

    rescored = {}
    for slice_y, item in detector_boxes_per_slice.items():
        p_t = p_by_slice.get(slice_y, 1.0)   # if missing, no rescaling
        rescored[slice_y] = {"boxes": item["boxes"], "scores": item["scores"] * p_t}
    return rescored
```

Component 7's `--use-gru` flag toggles whether `rescore_detector_outputs` is called before WBF aggregation.

---

## 9. CLI

```bash
# Phase 1: extract features for all 5 folds (sequential)
for f in 0 1 2 3 4; do
    uv run python -m src.gru_feature_cache --fold $f \
        --detector-ckpt runs/baseline_fold${f}/ckpts/best.ckpt \
        --cache-root /scratch/.../cache/v1
done

# Phase 2: train GRU for all 5 folds
for f in 0 1 2 3 4; do
    uv run python train_gru.py --fold $f \
        --feature-cache cache/v1/gru_features/fold${f} \
        --output-ckpt cache/v1/gru_ckpts/fold${f}.pt
done
```

---

## 10. Test plan

Tests in `tests/gru/`. Light by design.

### 10.1 Unit tests

| # | Test | Assertion |
|---|---|---|
| G1 | `test_gru_forward_shape` | Input (4, 50, 768) → output (4, 50) |
| G2 | `test_gru_bidirectional_uses_both_directions` | Ablate forward direction; assert outputs change |
| G3 | `test_volume_score_max_and_topk` | Synthetic per-slice probs with known max + top-5 → matches manual computation |
| G4 | `test_volume_score_respects_mask` | Padding values don't influence max or top-k |
| G5 | `test_collate_pads_correctly` | Batch with sequences of length [10, 20, 15]; output shape (3, 20, 768) with mask |
| G6 | `test_rescore_multiplies_scores` | Mock GRU returning constant p_t=0.5; assert all box scores halved |
| G7 | `test_rescore_handles_missing_slice` | Slice not in feature cache → score unchanged |

### 10.2 Integration tests

| # | Test | Assertion |
|---|---|---|
| G8 | `test_extract_features_for_fold_real` | Run on real fold-0 detector ckpt, write 3 patient .npz files; verify shape and dtype |
| G9 | `test_train_gru_synthetic_correlation` | Synthetic dataset where vol label correlates with feature signal; train 5 epochs; val AUROC > 0.7 |
| G10 | `test_train_gru_for_fold_e2e` | End-to-end on real fold-0 features; checkpoint saved; val AUROC computed |

### 10.3 Acceptance gate

Before Component 7 final eval can use rescoring:

1. All §10.1 unit tests pass.
2. All §10.2 integration tests pass.
3. All 5 feature caches built (`cache/v1/gru_features/fold{0..4}/` each contains correct number of .npz files).
4. All 5 GRU checkpoints trained (`cache/v1/gru_ckpts/fold{0..4}.pt`).
5. `gru_provenance.json` exists with per-fold val AUROC; **AUROC must be ≥ 0.5** for every fold (sanity floor — anything lower means the GRU is broken or features are useless).

---

## 11. Logging

Per fold (during GRU training):
- `gru_train/fold{f}/loss_max`, `loss_topk`, `loss_total`
- `gru_val/fold{f}/auroc`
- `gru_val/fold{f}/best_auroc`

Per fold (after feature extraction):
- `gru_features/fold{f}/n_patients`, `n_slices_total`, `extraction_seconds`

---

## 12. Failure modes

| Failure | Detection | Action |
|---|---|---|
| GRU val AUROC < 0.5 | per-fold gate | Hard-fail; investigate feature cache and GRU init |
| Feature cache missing slices | shape mismatch downstream | Re-run feature extraction for that patient |
| Detector ckpt EMA state missing | ckpt loader | Use live weights with warning; flag in provenance |
| OOM during feature extraction | torch error | Reduce batch_size to 8; backbone fwd is light, should be fine |

---

## 13. Wall-clock budget

- Feature extraction per fold: ~5 min (486 vols × 150 slices × backbone-only fwd).
- GRU training per fold: ~3 min (20 epochs × ~30 patient-batches).
- Total per fold: ~8 min.
- All 5 folds sequential: **~40 min total**.

---

## 14. Acceptance checklist (Component 6.5 done)

- [ ] `src/gru_feature_cache.py`, `src/gru_rescorer.py`, `train_gru.py` exist with the APIs in §5–§8.
- [ ] All §10.1 unit tests pass.
- [ ] All §10.2 integration tests pass.
- [ ] All 5 fold feature caches built and loadable.
- [ ] All 5 GRU checkpoints trained with val AUROC ≥ 0.5.
- [ ] `rescore_detector_outputs` callable from Component 7.
- [ ] `gru_provenance.json` written with per-fold metrics.

When this checklist is green, Component 7 (post-training evaluation) can begin.
