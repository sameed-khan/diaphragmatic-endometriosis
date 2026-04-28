# 2026 Modeling Synthesis: Diaphragmatic Endometriosis Detection

**Date:** 2026-04-27
**Audience:** Project lead deciding on the 2.5D detection pipeline
**Companion to:** `research_medical_imaging_approaches.md` (does not repeat material covered there)

This document is opinionated. Where I commit to a recommendation, I say so; where the literature genuinely doesn't tell us, I flag it as a guess.

---

## 1. 2.5D vs 3D detection in 2025–2026 for small lesions in anisotropic MR

**Consensus has not flipped since RSNA 2023.** Across the 2024–2026 literature, 2.5D + sequence rescoring continues to dominate when (a) data is small (<~500 positives), (b) voxels are anisotropic with slice thickness >2× in-plane spacing, and (c) lesions span only 1–2 slices. The 2024 brain-metastases comparison (Frontiers in Neuroinformatics) is illustrative: 2.5D detected 79% of metastases vs. 71% for the 3D model on highly anisotropic clinical MR; the 3D model only "won" on FP/case ([Hsu et al. 2022](https://www.frontiersin.org/articles/10.3389/fninf.2022.1056068/full)). The 2025 Nature npj Precision Oncology brain-tumor review reaffirms that pure-3D is mostly worth the cost only when the ROI is already cropped and isotropic ([Lopez-Larrubia 2025](https://www.nature.com/articles/s41698-024-00789-2)).

What has changed in 2024–2026 is mainly the *promptable foundation models*:

- **MedSAM-2** (April 2025): SAM-2 fine-tuned on 455k 3D image–mask pairs and 76k frames. It treats a volume as a video and propagates a single bbox prompt across slices. This is **not a replacement for a detector** — it still needs a prompt. But it's a strong post-detection refiner if you want lesion masks for free, and a strong annotation accelerator for the 22 holdout positives. ([Ma et al. 2025](https://arxiv.org/abs/2504.03600))
- **Discrepancy-based diffusion models** for anomaly localization (DDMD, 2024) get cited a lot but are mostly unsupervised and tested on BraTS-scale data. With 86 training positives and well-defined supervised labels, this is the wrong tool. Skip. ([Bercea et al. 2024](https://pubmed.ncbi.nlm.nih.gov/39217963/))
- **DETR variants in medical 2.5D**: DINO-DETR has been shown to match or beat Retina-U-Net on 4 medical datasets including LIDC and KiTS19 ([Baumgartner et al. 2023, MICCAI Workshop](https://link.springer.com/chapter/10.1007/978-3-658-41657-7_39)). For our scale, this is interesting but not transformative.

**My call:** stay 2.5D. 3D is only justified after a 2.5D baseline is in place and you are FP-limited rather than recall-limited. The slice-thickness anisotropy (3–6.9mm vs 0.7mm in-plane, ratio 4–10×) makes 3D convolutions especially poorly matched to this data.

---

## 2. Detection head choice

For ~108 positives with 1–10 lesions each, and most lesions < 30mm:

**Ranked recommendation:**

1. **RTMDet-S/M (anchor-free, dense head)** — *primary pick*. RTMDet-tiny gets 41.1 AP COCO at 4.8M params, RTMDet-S 44.6 AP at 8.9M. The depthwise large-kernel design and SimOTA dynamic label assignment specifically helped small/medium objects in the 2025 brain-tumor RTMDet study ([Mehmood et al. 2025, Bioengineering](https://www.mdpi.com/2306-5354/12/3/274)). For a small-data medical regime, anchor-free + dynamic assignment avoids the manual anchor tuning RetinaNet/Faster R-CNN demand, and the model is small enough to not overfit 86 training positives.
2. **YOLOv11-s** — *strong fallback*. Won the 2025 mammography lesion benchmark on Jetson, and its C2PSA/SPPF blocks are explicitly tuned for small dense targets. Tooling (Ultralytics) is mature and ergonomic. ([Ultralytics docs](https://docs.ultralytics.com/compare/yolo11-vs-rtdetr/), [mammography study 2025](https://www.mdpi.com/2072-6694/18/1/70))
3. **RT-DETR-R18 / RTDETRv2** — *try if 1 and 2 plateau*. Wins on dense/sub-pixel targets (e.g., diabetic-retinopathy microaneurysms) and dental pathology mAP@50 ([2025 dental study](https://www.sciencedirect.com/science/article/pii/S2215016125005400)). NMS-free is genuinely useful when lesions cluster. Downside: ~3× CUDA memory at training, slower convergence — painful with 86 positives in 5-fold CV.
4. **Co-DETR / DINO-DETR** — *if you have spare compute and time*. Co-DETR's auxiliary one-to-many heads partially solve DETR's "sparse positive supervision" problem, which is exactly what bites in a tiny-positive regime. Worth a fold or two of validation, but I'd not stake a 2-week timeline on it.
5. **RetinaNet, Faster R-CNN, anchor-based YOLO (v8/v9)** — solid baselines but anchor design is annoying and the modern anchor-free heads simply do better for small medical objects in 2024–2026 small-object surveys ([MDPI 2025 review](https://www.mdpi.com/2076-3417/15/22/11882)).

**Anchor vs anchor-free, query vs dense:** anchor-free dense heads (RTMDet, YOLOv11) currently dominate small-object medical work in the 100–1000 positives regime. Query-based (DETR family) becomes more attractive at >1000 positives because the bipartite-matching loss is data-hungry. We are well below that threshold.

---

## 3. Backbone choice (priority order)

Given 2.5D 3-channel input, ~86 training positives per fold, and ImageNet/RadImageNet pretraining:

1. **ConvNeXt-tiny (28M)** — *primary*. Won/co-won RSNA 2023 abdomen, 2024 mammography (ConvNeXt-small variant), and 2025 multi-modal medical classification benchmarks. Strong inductive biases, plays well with small data, has stable training dynamics, and ImageNet-22k weights are widely available. ConvNeXt-V2 with FCMAE pretraining gives a small extra bump but is also more finicky. ([CVPR 2023 ConvNeXt-V2](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf))
2. **EfficientNetV2-S (22M)** — *fallback*. Used by RSNA 2023 1st place (Team Oxygen). Slightly less peak mAP than ConvNeXt-tiny in small-object detection but trains fast and is parameter-efficient for our data scale.
3. **SE-ResNeXt50 (28M)** — *conservative baseline*. Veteran of RSNA 2019/2022 winners. If ConvNeXt-tiny overfits, fall back here. SE blocks help small-target sensitivity.
4. **MaxViT-tiny (31M)** — *experimental*. Hybrid conv-transformer that pairs well with ConvNeXt-V2 in ensembles ([CXR-LT 2024 4th place](https://arxiv.org/abs/2410.10710)). I'd only deploy in late ensemble.
5. **RAD-DINO ViT-B (86M)** — *opportunistic*. Microsoft's released DINOv2-style chest-X-ray foundation backbone. **Caveat: trained on CXR, not abdominal MR.** Domain mismatch is real; I would not stake the primary model on this. Worth a single ablation at the end. ([HF model card](https://huggingface.co/microsoft/rad-dino))
6. **BiomedCLIP** — designed for vision-language, not dense prediction. Skip for detection unless you specifically want zero-shot.

**Honest uncertainty:** at 86 positives, the gap between ConvNeXt-tiny and EfficientNetV2-S will likely be within noise. The cross-validation variance from a single bad fold split is plausibly larger than the architecture effect.

---

## 4. Pretraining options

The 2024 RadImageNet replication confirms a **~4–5% AUC bump** for MR-specific tasks at small data (ACL/meniscus tears, n<1000) over ImageNet pretraining ([Mei et al., RadImageNet 2022/2024 follow-ups](https://pubs.rsna.org/doi/full/10.1148/ryai.220126)). For abdominal-MR detection at our scale, this bump is plausibly worth the integration effort.

**MR foundation models in 2024–2026 — honest take:**

- **No abdominal-MR-specific foundation model exists yet** that has the scale + open weights that you'd want. MerLin is CT only ([Nature 2025](https://www.nature.com/articles/s41586-026-10181-8)). USFM is ultrasound only ([USFM 2024](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001270)). RAD-DINO is CXR only.
- **MerMED-FM** (mid-2025, arxiv:2507.00185) is multimodal but does not include abdominal MR meaningfully and weights/availability are unclear.
- The **RadImageNet** weights remain the best practical "MR-aware" generic backbone for 2D conv backbones at small data. They cover ~672k MR slices including abdominal.

**Recommendation, ranked:**

1. ImageNet-22k → ConvNeXt-tiny → fine-tune (always your baseline)
2. RadImageNet → ResNet50 / DenseNet → fine-tune (run as a parallel ablation; gain is real but modest)
3. Self-supervised in-domain pretraining on the 500 negatives + unlabelled slices via DINOv2 or MAE — *only if you have spare GPU-weeks*. With 500 volumes ≈ 60k slices, this is barely above the data scale where SSL helps.
4. Skip BiomedCLIP / RAD-DINO for primary detector.

---

## 5. Slice-stack channel design

I'll commit to a specific recipe.

**Primary:** **3 adjacent slices, stride 1** (center, ±1). Maps directly onto pretrained ImageNet RGB conv1, no first-conv surgery. At 3.0–3.6mm typical slice thickness, the receptive field is ~9–11mm of through-plane context — enough for nodules but not so much that micronodules average out.

**Secondary ablation:** **5 adjacent slices, stride 1**, with `conv1` modified by replicating the 3-channel pretrained weights and re-normalizing. RSNA 2022 1st place used exactly this (5 + mask = 6 channels) ([Pan 1st place writeup](https://github.com/i-pan/kaggle-rsna-cspine)).

**Don't:**
- Use stride > 1 with 3.0–3.6mm slices: a micronodule (≤5mm) at 3mm thickness is 1–2 slices wide, so stride-2 sampling will alias it out half the time. The literature on this is sparse but the geometry is unambiguous.
- Encode the liver mask as an RGB-replacing channel. **Do** consider it as an *added* 4th channel only if you also adapt conv1. Whether a binary liver-mask channel actually helps a detector that has *already been cropped to that mask + 20mm dilation* is unclear — the information is largely redundant. I would skip it in v1.
- Add a sinusoidal positional-encoding channel for slice index. Cute idea, no published evidence it helps detection, and it can leak slice-position shortcuts (see §12).

**Genuine uncertainty:** does CSA-Net's cross-slice attention beat plain channel-stacking here? On their multi-task benchmarks it does ([CSA-Net 2024](https://github.com/mirthAI/CSA-Net)), but those were segmentation tasks. For detection of 1–2 slice lesions, the attention has very few inter-slice relations to model. I'd defer this.

---

## 6. Anchor / query priors for tiny lesions

At 0.7 mm/pixel in-plane, lesions of 5–30 mm are **7–43 px** on the native grid. After cropping to a 20mm-dilated liver ROI and padding, in-plane pixel size should remain ~0.7mm.

**For an anchor-free head (RTMDet, YOLOv11)** the equivalent decision is FPN levels and stride choice:
- Use **strides {8, 16, 32}** (skip stride-64 — useless for our scale).
- The smallest detectable scale at stride-8 with kernel ~3×3 is ~24 px = 17mm. To detect 5–10mm micronodules, **add a stride-4 (P2) head**. This is the same trick the 2025 small-object surveys recommend for sub-15px targets ([MDPI 2025 SOD review](https://www.mdpi.com/2076-3417/15/22/11882), [FocusDet 2024](https://www.nature.com/articles/s41598-024-61136-w)).

**For an anchor-based head (RetinaNet/Faster R-CNN)**, set anchors by k-means on your training boxes. As a starting point given 5–30mm range:
- Scales: **{8, 16, 32, 64} px** at stride-4 (P2 level). At stride-8 add **{16, 32, 64, 128}**.
- Aspect ratios: **{0.7, 1.0, 1.4}**. Lesions are not strongly elongated; anything wilder is overfitting.
- Run nnDetection's auto-anchor evolutionary search if you go nnDetection ([Baumgartner 2021](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_51)). It will likely converge on roughly the above.

**For DETR variants:** use ~300 queries (DINO-DETR default). Don't reduce below 100 or you'll drop the rare multi-lesion cases.

---

## 7. Class-imbalance handling

The patient-level imbalance (5:1 negative:positive) is benign. The slice-level imbalance is the problem: in 86 positive volumes × ~120 slices, with lesions on 1–2 slices each, *positive-slice fraction is ~1–3%*; with 500 negative volumes, it drops below 0.5%.

**Concrete recipe:**

1. **Slice sampling, not slice listing.** During training, draw batches with a fixed positive-slice fraction (start at 0.5; decay to 0.25 across epochs). This is the medical-imaging analog of the foreground oversampling nnU-Net does by default and what every RSNA winner has done since 2019.
2. **Use the 500 negatives as hard-negative mining sources, not equal-weight training data.** After 1 warmup epoch, score all negative-volume slices with the current model and oversample top-k FP-prone slices for the next epoch. This is the OHEM idea, slice-level. Empirically helped in RSNA mammography 1st place.
3. **Loss for classification head: focal loss, γ=1.5, α=0.25.** The 2024 small-object survey converges on γ=1.5–2.0 for tiny medical targets; γ=2.0 is the YOLO/RetinaNet default but can over-down-weight easy positives when positives are scarce ([learnopencv on focal/SIoU](https://learnopencv.com/yolo-loss-function-siou-focal-loss/)). For DETR-family, **varifocal loss** is preferable since it weights by IoU quality ([VarifocalNet CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf)).
4. **Don't use OHEM at the proposal level for two-stage detectors.** With <10 GT boxes per volume it tends to instability. If using Faster R-CNN, use the built-in 1:3 fg/bg ratio sampler.

---

## 8. Augmentation strategy

Right-side bias is real and load-bearing here (Revel et al. 2016: 87.5% of diaphragmatic endometriosis is right-sided). Horizontal flips would destroy this prior.

**Use:**
- Small affine: rotation ±10°, scale 0.9–1.1, translation ±5%. Don't go bigger — you'll move lesions outside the dilated liver crop.
- Intensity: γ ∈ [0.8, 1.2], multiplicative bias 0.9–1.1, Gaussian noise σ=0.01 on normalized intensity.
- Light elastic (σ=2, control points ~8). Has a track record in BraTS/abdomen segmentation winners.
- **Lesion copy-paste augmentation**, restricted to the right hemidiaphragm region. The 2024 LAMA paper for skin lesions and Cut Instance Mixing for GI lesions both showed clean wins over CutMix/MixUp in medical-detection settings ([LAMA 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300415/), [CIM 2026](https://www.nature.com/articles/s41598-026-42138-2)). With only 86 positives this is one of the highest-leverage interventions on the list.

**Avoid:**
- Horizontal flip (breaks the right-side prior).
- Vertical flip (anatomically wrong).
- Cutout/CutMix on whole images (can erase the only lesion).
- Mixup on detection (it works for classification, but for box regression on tiny targets it muddies localization; the 2024–2025 MediAug benchmark found mixed results across mix-augmentations ([MediAug 2025](https://arxiv.org/html/2504.18983v1))).
- Mosaic (YOLO default): plausible *if* you restrict to 2×2 of right-hemidiaphragm crops. For v1, just disable it. Mosaic + small medical datasets has burned multiple Kaggle teams.

---

## 9. Loss function

**Box regression:** **CIoU**. SIoU and DIoU are marginal improvements in remote-sensing benchmarks but the 2024 YOLO-loss reviews show CIoU remains within noise on most tasks ([learnopencv GFL/VFL post](https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/)). CIoU is the safer default.

**Classification:** focal loss as above (γ=1.5, α=0.25), or **varifocal loss** if you go RT-DETR/DINO.

**Auxiliary segmentation head**: yes, add it. Multi-task with the existing 3D segmentation masks is one of the most reliable ways to inject regularization at small sample size. RSNA 2023 1st place did this. Use **Dice + BCE on the center slice only**, weight 0.3 vs the detection loss. The mask gives the network a denser supervision signal where it currently has only 4 numbers per box.

**Don't** use blob loss / CC-DiceCE / CATMIL on the segmentation aux head. They're tuned for *primary* segmentation; here the segmentation is auxiliary regularization, not the metric you optimize.

---

## 10. CV evaluation strategy

86 CV positives across 5 folds = ~17 positive validation volumes per fold. This is small.

**Concrete plan:**

1. **Primary metric: FROC at 1, 2, and 4 FP/volume.** This is the standard for medical detection and the only one your radiology collaborators will trust ([2024 CAD metrics review](https://www.mdpi.com/2306-5354/11/11/1165)). Operating point selection: I'd ask for sensitivity at **2 FP/volume** as the headline (Lunit blog convention; LIDC-IDRI uses 0.125, 0.25, 0.5, 1, 2, 4, 8 — report all and bold 2).
2. **Secondary: AP@IoU=0.3** (yes, 0.3 — endometriosis lesions are small enough that IoU=0.5 is too strict and noisy; ask anyone who tried mAP on PI-CAI). Also report PR-AUC.
3. **Confidence intervals: bootstrap by patient.** Per fold, draw 1000 patient-level bootstrap samples of the validation set, recompute sensitivity-at-FPR=2/volume, report the 2.5th–97.5th percentile. Aggregate across folds with the **CPM (Competition Performance Metric)**: mean sensitivity at {0.125, 0.25, 0.5, 1, 2, 4, 8} FP/volume. This is exactly what LUNA16 and PI-CAI did.
4. **Fold split must be patient-level, stratified by lesion count and lesion-size bin.** Stratified by lesion size matters because a single 30mm plaque case in a fold dominates AP.
5. **Don't compute volume-level AUC as your primary metric.** With 17 positives and 100 negatives per fold, AUC has CI of roughly ±0.08; you cannot tell two models apart on it.
6. **Hold the 122 holdout volumes truly held out.** Touch them once, after CV is locked.

---

## 11. Inference-time slice aggregation

Pipeline I'd commit to:

1. **Per-slice 2D detection** with 3-slice context input. For each volume, you get a sequence of (slice_idx, box, score) triples.
2. **3D NMS via WBF in (x, y, z).** Treat the slice index as the z coordinate and run **Weighted Box Fusion 3D** ([ZFTurbo's WBF repo](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)), IoU threshold 0.3 in xy, and require ≥2 adjacent slices to agree for non-micronodule lesions. For micronodules (≤5mm = ≤2 slices), accept single-slice detections but with a higher score threshold. WBF beat NMS and Soft-NMS in lung disease detection ([Solovyev et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0262885621000226)).
3. **Sequence-model rescoring (optional Phase-2):** GAP-pool the 2D feature maps at each slice, feed the (z, embedding) sequence to a small bidirectional GRU (1 layer, hidden 128), and add a "lesion present in this volume?" output that supervises with the volume label. Use the GRU's per-slice presence probability to rescore each box's confidence: `score' = score × σ(GRU_z)`. This is the RSNA 2019 ICH and RSNA 2023 abdomen pattern, applied to box rescoring. ([RSNA 2019 2nd place](https://github.com/darraghdog/rsna))

Skip 3D NMS implemented as a literal IoU-3D — your slice resolution is too coarse for IoU-3D to be well-defined for 1–2-slice lesions.

---

## 12. Concrete pitfalls

1. **TotalSegmentator MRI false negatives at the dome.** TS-MRI was trained on whole-body data with median in-plane resolution that is finer than your coronal Dixon; the [TS-MRI paper](https://pubs.rsna.org/doi/10.1148/radiol.241613) explicitly notes failure modes when section thickness >6mm and contrast is low at organ boundaries — both apply here. The diaphragmatic dome is precisely where TS-MRI tends to under-segment because the liver–lung interface has minimal MR contrast on water-only Dixon. **Mitigation:** the 20mm dilation you've chosen is exactly the right safety margin — but **don't trust the dilation to fully cover all positives**. Run a one-time audit: for every positive volume, check that all lesion-mask voxels lie inside the dilated ROI. If any are outside, increase the superior-direction dilation specifically, or mirror across the diaphragm before dilating.

2. **FP from bowel, kidney, adrenal, spleen.** All four organs are partly inside a 20mm-dilated liver mask in coronal Dixon, and all four can produce small bright structures on water-only Dixon (collapsed bowel, renal cortex enhancement, splenic flexure). Mitigation: include the organ-class as an auxiliary multi-class segmentation channel during training, or use TS-MRI to *erode out* the kidney/spleen voxels from the ROI. Most likely culprit for FPs in v1: posterior bowel near hepatic flexure.

3. **Slice-thickness shortcut learning.** Your slice thickness varies 2.2–6.9mm. Across a single GE 1.5T scanner this is mostly protocol-driven, but it can correlate with year of acquisition, indication, and radiologist preference, which can correlate with disease prior probability. The 2024 [scanner-domain-shift study](https://arxiv.org/html/2409.04368v2) shows MR is the most domain-shift-sensitive modality. **Mitigation:** during EDA, plot slice thickness distribution by label and by fold. If positives skew toward thinner slices (likely — symptomatic patients get the better protocol), report a thickness-stratified FROC. Also: **resample everything to a common through-plane spacing** (3.0mm is a reasonable target) before slice stacking, otherwise the channel triplet means different physical context per case.

4. **Liver-edge detection-as-shortcut.** With cropping aligned to the liver mask, the network can learn "the bright dome edge is suspicious." This is fine when the dome is correctly outlined and dangerous when it isn't. Mitigation: during training, augment the cropping ROI by ±5mm random dilation jitter so the dome isn't always at the same pixel.

5. **5-fold CV variance with 17 positives/fold.** A single mis-stratified fold can swing your headline number by 5+ AP points. **Run 3 random seeds per fold** and average — this is cheaper than going to 10-fold CV and gives you variance estimates.

6. **Annotation leakage between adjacent slices.** When the same lesion is annotated on slices z and z+1, your "positive" 3-slice stacks centered on z, z+1, z+2 share GT. Bookkeeping issue: when computing slice-level metrics (not your primary, but useful for debugging), make sure positive-slice clustering is by lesion ID, not by slice ID. Doesn't affect FROC.

---

## If I had only 2 weeks of compute, here is the plan

**Budget:** assume 4× A100 80GB or equivalent.

### Week 1 — single hard-baked baseline

- **Model:** RTMDet-S, ConvNeXt-tiny backbone (ImageNet-22k weights via timm), 3-slice channel input (center ±1), 512×512 input after 20mm-dilated liver crop and pad-to-square.
- **Heads:** stride-{4, 8, 16, 32}; the stride-4 P2 head is the non-default but mandatory addition.
- **Auxiliary segmentation head:** Dice+BCE on center slice, weight 0.3.
- **Loss:** focal (γ=1.5, α=0.25) + CIoU.
- **Augmentation:** affine (rot ±10°, scale ±10%, trans ±5%), γ ∈ [0.8, 1.2], Gaussian noise σ=0.01, light elastic (σ=2). **No** flips, **no** mosaic, **no** mixup.
- **Sampling:** per-batch, 50% positive slices in epoch 0–10, decaying to 25% by epoch 30. Negative volumes provide 50% of negative slices initially; after epoch 5, 30% of those come from a hard-negative pool refreshed each epoch.
- **Schedule:** AdamW, lr 2e-4, cosine to 1e-6, weight decay 0.05, 60 epochs, EMA decay 0.999.
- **CV:** the frozen 5-fold split, 3 seeds per fold = 15 runs. Each run ≈ 4–5 hours on 1× A100 → 60–75 GPU-hours total. Easily fits in week 1 on 4 GPUs.
- **Inference:** TTA = none in v1 (just adds noise at this scale). 3D WBF aggregation, IoU=0.3, allow single-slice for boxes ≤5mm, require 2 adjacent for larger.
- **Headline metric:** sensitivity at 2 FP/volume + CPM, with patient-bootstrap 95% CI.

### Week 2 — three targeted experiments

Run these three in parallel; each is one 5-fold sweep (~25 GPU-hours):

1. **Lesion copy-paste augmentation (right-hemidiaphragm only).** Highest-EV intervention given 86 positives. If it adds >2 points sensitivity-at-2-FP, lock it in.
2. **5-channel slice stack (center ±2)** with replicated-and-renormalized conv1. Tests whether through-plane context buys anything beyond 3 channels at this slice thickness.
3. **GRU rescoring head** on top of the frozen Week-1 detector. Train only the GRU on patient labels for 20 epochs. If volume-level AUC and per-volume sensitivity both improve, ship it.

### What I would *not* spend Week 2 on

- 3D detection (nnDetection, etc.). Will probably tie or lose.
- RT-DETR or DINO-DETR. Better baseline first.
- RAD-DINO / BiomedCLIP backbones. Domain mismatch + ViT first-conv surgery is a Week 3 project.
- MedSAM-2 in the loop. It's an annotation tool, not a detector.

### What would change my mind

- If Week 1 baseline gets <40% sensitivity at 2 FP/volume across folds, the bottleneck is data/labels, not model. Reinvest in label QA and copy-paste augmentation, not architecture.
- If Week 1 baseline gets >70% sensitivity at 2 FP/volume, ensemble RTMDet + YOLOv11 + a single DETR variant for the holdout submission. Diminishing returns on architecture beyond that.

---

## Sources

- [Hsu 2022, 2.5D vs 3D brain mets](https://www.frontiersin.org/articles/10.3389/fninf.2022.1056068/full)
- [Lopez-Larrubia 2025, npj Precision Oncology brain tumor DL review](https://www.nature.com/articles/s41698-024-00789-2)
- [MedSAM2 2025](https://arxiv.org/abs/2504.03600), [code](https://github.com/bowang-lab/MedSAM2)
- [Bercea et al. 2024, discrepancy diffusion brain MRI](https://pubmed.ncbi.nlm.nih.gov/39217963/)
- [DETR variants on medical 4-dataset benchmark](https://link.springer.com/chapter/10.1007/978-3-658-41657-7_39)
- [RTMDet 2022 paper](https://arxiv.org/abs/2212.07784); [RTMDet brain tumor 2025](https://www.mdpi.com/2306-5354/12/3/274)
- [YOLOv11 vs RT-DETR mammography 2025](https://www.mdpi.com/2072-6694/18/1/70); [dental pathology 2025](https://www.sciencedirect.com/science/article/pii/S2215016125005400)
- [RT-DETR medical imaging 2025](https://arxiv.org/abs/2501.16469)
- [ConvNeXt V2 CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf)
- [ConvNeXt V2 + MaxViT CXR-LT 2024](https://arxiv.org/abs/2410.10710)
- [RAD-DINO Hugging Face](https://huggingface.co/microsoft/rad-dino)
- [BiomedCLIP](https://arxiv.org/html/2303.00915v2/)
- [RadImageNet 2022](https://pubs.rsna.org/doi/full/10.1148/ryai.220126)
- [Merlin CT vision-language 2025](https://www.nature.com/articles/s41586-026-10181-8); [USFM 2024](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001270); [MerMED-FM 2025](https://arxiv.org/abs/2507.00185)
- [LAMA lesion-aware mixup 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300415/); [Cut Instance Mixing 2026](https://www.nature.com/articles/s41598-026-42138-2); [MediAug 2025](https://arxiv.org/html/2504.18983v1)
- [VarifocalNet CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf); [SIoU/Focal/GFL learnopencv](https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/)
- [Small Object Detection 2023–2025 review](https://www.mdpi.com/2076-3417/15/22/11882); [FocusDet 2024](https://www.nature.com/articles/s41598-024-61136-w)
- [CAD performance metrics review 2024](https://www.mdpi.com/2306-5354/11/11/1165); [Lunit FROC blog](https://medium.com/lunit/evaluation-curves-for-object-detection-algorithms-in-medical-images-4b083fddce6e)
- [Weighted Boxes Fusion paper](https://www.sciencedirect.com/science/article/abs/pii/S0262885621000226); [WBF repo](https://github.com/ZFTurbo/Weighted-Boxes-Fusion); [WBF lung disease 2021](https://www.researchgate.net/publication/354151389_Weighted_Box_Fusion_Ensembling_for_Lung_Disease_Detection)
- [TotalSegmentator MRI 2025](https://pubs.rsna.org/doi/10.1148/radiol.241613)
- [Scanner domain shift in medical DL 2024](https://arxiv.org/html/2409.04368v2)
- [nnDetection 2021](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_51); [aneurysm MRI](https://arxiv.org/abs/2305.13398)
- [PI-CAI Grand Challenge](https://pi-cai.grand-challenge.org/); [PI-CAI Lancet Oncol 2024](https://pubmed.ncbi.nlm.nih.gov/38876123/)
