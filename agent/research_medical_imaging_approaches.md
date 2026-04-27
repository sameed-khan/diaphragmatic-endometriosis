# Medical Imaging Object Detection/Segmentation Research

## For: Diaphragmatic Endometriosis Lesion Detection in 3D MRI

**Date:** 2026-04-25  
**Context:** ~131 positive volumes with segmentation masks + ~1000 negatives. Lesions are very small, bright spots on fat-suppressed T1-weighted MRI, appearing on the superior/medioinferior border of the liver near the hepatorenal space.

---

## 1. 2.5D vs 3D Approaches: What Won

### The Dominant Pattern Across RSNA Competitions

A remarkably consistent pattern emerges across all major RSNA Kaggle competitions: **2.5D approaches (2D CNN + sequence model) consistently outperform pure 3D CNNs**, especially when training data is limited.

#### RSNA 2019 Intracranial Hemorrhage Detection
- **1st place (SeuTao):** Ensemble of 2D CNN classifiers (SE-ResNeXt101, DenseNet169, DenseNet121 at 256-512px) + sequence models. Three-window channel strategy (brain, subdural, bone windows as RGB channels).
- **2nd place (NoBrainer):** Single 2D image classifier (ResNeXt at 480px with windowing) -> extract GAP embeddings -> feed into LSTM. Key insight: *"We have a single image classifier, where data is split on 5 folds but only trained on 3 of them. We then extract the GAP layer from the classifier with TTA and feed into an LSTM."*
- **All top-5 solutions** used 2D CNNs + sequential models (Bidirectional GRU or LSTM). Number of models in ensembles ranged from 7 to 31.
- A later Transformer-based approach beat the 1st place solution using only 25% of the parameters and 10% of the FLOPs.

**Key takeaway:** The 2-stage paradigm (2D feature extraction -> sequence aggregation) was universally dominant.

#### RSNA 2022 Cervical Spine Fracture Detection
- **1st place:** 2-stage: (1) 3D semantic segmentation (ResNet18/EfficientNetV2s + UNet, 128x128x128) to locate vertebrae, (2) 2.5D classification with LSTM. For each vertebra, extracted 15 evenly-spaced slices, then took +/-2 adjacent slices to form **5-channel 2.5D images**. Added predicted segmentation mask as 6th channel. *"Simple 3D CNN doesn't work."*
- **3rd place (darraghdog):** 2.5D CNN + 1D RNN on random overlapping z-axis slices. Backbones: `resnest50d` and `seresnext50_32x4d`. Used study-level bounding box to crop all slices.
- **5th place:** 3-stage: (1) 2D classification/segmentation for vertebrae, (2) multiply with fracture labels for slice-level labels, (3) aggregation model for study-level predictions.
- **6th place (Ian Pan):** Combined 3D CNN classification (X3D models) on cropped vertebra chunks with a "TD CNN" (temporal/depth CNN). Extracted 3D features and fused them. This was one of the few competitive 3D approaches but still required segmentation-based cropping first.

**Key takeaway:** Even with 3D segmentation available, top solutions used it primarily as a *preprocessing step* to crop/localize ROIs, then applied 2.5D classification.

#### RSNA 2023 Abdominal Trauma Detection
- **1st place (Team Oxygen):** 3-part solution: (1) 3D segmentation for masks/crops, (2) 2D CNN + RNN for organ classification, (3) 2D CNN + RNN for bowel/extravasation. Backbones: **CoaT Lite Medium, EfficientNetV2s + GRU**. Input: crop volumes to 96x384x384, then reshape to 32x3x384x384 (32 2.5D triplets). Multi-task: segmentation + classification. Used predicted organ masks as soft-label targets.
- **2nd place (TheoViel):** 2D CNN + RNN. Key innovation: **crop models** for kidney/liver/spleen using 3D ResNet18 for organ localization, then feed crops to 2D CNN + RNN. Backbones: ConvNeXt-v2, MaxViT, CoatNet. *"2D + crop model improved kidney/liver/spleen by a good margin."* Final ensemble: 3x 2D models + 8x 2.5D models.
- **10th place:** Segmentation -> Crop -> separate organ models -> stacking.

**Key takeaway:** ROI cropping using 3D segmentation + 2.5D classification is the winning formula for abdominal imaging.

#### RSNA 2024 Lumbar Spine Degenerative Classification
- **2nd place:** Multi-stage: (1) axial level estimation, (2) sagittal slice estimation, (3) region estimation by YOLOX, (4) 2.5D classification. Used `model.arch=2.5d` with `in_channels=3`, 224px image size, AdamW optimizer, lr=2e-3, EMA with decay 0.99, Mixup=0.3.
- **4th place:** Explicit 2.5D architecture with attention and horizontal flip augmentation.
- **Silver medal solutions:** YOLOv8-based object detection approach.

#### RSNA Screening Mammography (2023)
- **1st place:** YOLOX for ROI detection -> ConvNeXt-small for classification. 4-fold patient-level splits. Key was aggressive ROI cropping.

### Summary: When Does 3D Win?

3D models are competitive **only** when:
1. Used for coarse localization/segmentation (to generate crops for 2.5D)
2. The ROI has already been tightly cropped
3. There is abundant training data (>1000+ annotated volumes)

3D models lose because:
- They require much more memory -> smaller input resolution or batch sizes
- They can't leverage ImageNet pretrained weights
- They overfit with small datasets
- Anisotropic voxel spacing in many clinical MRI sequences hurts 3D convolutions

**For our task (131 positives):** 2.5D is strongly recommended. Pure 3D will likely overfit.

---

## 2. Backbones & Architectures for Small Object Detection

### What Worked in Competitions

| Competition | Top Backbones | Notes |
|---|---|---|
| RSNA 2019 ICH | SE-ResNeXt101, SE-ResNeXt50, DenseNet121/169 | Heavy backbones with SE attention |
| RSNA 2022 C-Spine | ResNet18 (for 3D seg), ResNeSt50d, SE-ResNeXt50 | Lighter backbones for 2.5D |
| RSNA 2023 Abdomen | CoaT Lite Medium, EfficientNetV2s, MaxViT, ConvNeXt-v2 | Transformer-hybrid architectures |
| RSNA 2024 Lumbar | YOLOv8, various 2.5D CNN | Object detection + classification |
| RSNA Mammography | ConvNeXt-small, EfficientNetV2S, SE-ResNeXt | Medium-sized efficient backbones |
| Endometriosis (paper) | 3D-DenseNet-121 | Used with multi-sequence MRI |

### Architecture Insights for Small Lesion Detection

1. **ResNet-50 is reasonable but not optimal.** SE-ResNeXt50 and EfficientNetV2s are more commonly used in winning solutions. ConvNeXt-small/tiny are the most modern choice.

2. **For 2.5D with 131 samples:** A lighter backbone is advisable to prevent overfitting. Consider:
   - EfficientNetV2-S or EfficientNet-B3/B4
   - ConvNeXt-tiny
   - ResNet-34 or ResNet-50 (not deeper)
   
3. **Attention mechanisms help:** SE blocks (Squeeze-and-Excitation) consistently appeared in winning solutions. CoaT (Co-Scale Conv-Attentional Transformer) was the 2023 favorite.

4. **For segmentation tasks specifically:** U-Net variants remain dominant. nnU-Net auto-configures well. For small lesion segmentation, Attention U-Net and SwinUNETR have been used but require fine-tuning to avoid collapse on small datasets.

### Our Consideration: Shallow ResNet-50

Using a shallow (early-truncated) ResNet-50 with pretrained weights is a *reasonable but conservative* choice. Based on the evidence:
- It will work but may not be optimal
- EfficientNetV2-S or ConvNeXt-tiny may be better choices for the backbone
- The pretrained weights advantage is real and important given our small dataset
- Consider SE-ResNet50 or ResNeXt50_32x4d for marginal improvement

---

## 3. Handling Class Imbalance & Tiny Lesion Detection

### Competition Approaches

1. **Positive/negative sampling ratio in patches (MONAI standard):** `RandCropByPosNegLabeld` - ensures balanced sampling of patches centered on lesions vs. background. Used in essentially all MONAI-based solutions.

2. **Oversampling positive cases:** In RSNA mammography (2% positive rate), winners oversampled cancers during training. Similarly applicable to our ~12% positive rate (131/1131).

3. **Multi-task learning with segmentation:** RSNA 2023 1st place trained segmentation + classification simultaneously. The segmentation branch provides direct supervision for lesion locations.

4. **Soft-labeling with organ visibility:** RSNA 2023 1st place used normalized organ mask areas across slices as soft targets, preventing hard threshold issues.

5. **ROI cropping as implicit class balancing:** By cropping to the liver/diaphragm region first, you dramatically reduce the background-to-lesion ratio.

### Loss Functions for Small Lesions (Ranked by Evidence)

#### Tier 1: Well-Established
1. **Dice + Cross-Entropy (DiceCE)** - nnU-Net default. Solid baseline. Dice handles class imbalance, CE provides stable gradients.
   
2. **Focal Loss** - Designed for hard-to-classify examples. γ=2.0 is standard; for very small lesions, γ=3.0 may help. α should weight the minority class.

3. **Dice + Focal Loss combination** - Common in BraTS challenge winners. Dice maintains structural accuracy, Focal targets difficult pixels.

#### Tier 2: Instance-Aware (Specifically for Multiple Small Lesions)
4. **Blob Loss** (Kofler et al., 2022) - Computes loss per-instance (connected component), then averages. Specifically addresses instance imbalance where large lesions dominate. Improved F1 by 5% for MS lesions, 3% for liver tumors. Available as nnU-Net plugin.

5. **CC-DiceCE** (2025) - Based on CC-Metrics framework. Increases recall for small lesions with minimal segmentation degradation. Outperforms blob loss in multi-dataset benchmarks within nnU-Net framework. Code: https://github.com/TIO-IKIM/Learning-to-Look-Closer

6. **CATMIL Loss** (2026) - Component-Adaptive Tversky + Multiple Instance Learning. Best balanced performance in recent benchmarks: Dice 0.7834, small lesion recall 0.8730 (vs. 0.7956 for DiceCE baseline, +7.74%). Code: https://github.com/luumsk/SmallLesionMRI

#### Tier 3: Other Notable Options
7. **Focal Tversky Loss** (Abraham & Khan, 2019) - Generalizes Dice with tunable α/β for precision/recall trade-off. Good for small lesions (tested on data where lesions occupy <5% of image area).

8. **Unified Focal Loss** (2021) - Framework that generalizes Dice and CE-based losses. Robust to class imbalance across multiple tasks. Code: https://github.com/mlyg/unified-focal-loss

9. **Inverse-Weighted Loss Reweighting** (2020) - Reweights loss by inverse lesion size. Specifically improves detection of small lesions in LUNA16 and liver metastasis datasets. Works as a wrapper around any base loss.

### Recommendation for Our Task

Start with **DiceCE** as baseline, then experiment with:
1. **Dice + Focal Loss** (most practical improvement)
2. **Blob loss** or **CC-DiceCE** if we have multiple lesion instances per volume
3. **CATMIL** if small lesion recall is our primary concern

---

## 4. Pre-processing Strategies

### Cropping
- **ROI cropping is universally used** in top solutions. Every RSNA competition winner pre-crops to the region of interest.
- RSNA 2023 1st place: 3D segmentation (ResNet18 U-Net) to locate organs -> crop to organ region -> feed to classifier
- RSNA 2022: Study-level bounding box from segmentation, crop all slices identically
- **For our task:** Pre-cropping around the liver/diaphragm region is strongly supported by the evidence. Consider:
  - Using a pretrained liver segmentation model (e.g., TotalSegmentator) to generate liver masks
  - Cropping to a fixed region around the right hemidiaphragm (where 87.5% of diaphragmatic endometriosis lesions occur according to Revel et al., 2016)

### Windowing/Normalization
- **CT competitions:** Multiple windowing is critical (brain/subdural/bone windows as channels). This is the single most impactful preprocessing for CT.
- **MRI:** Intensity normalization approaches differ:
  - Min-max normalization to [0, 1] - simplest, commonly used
  - Z-score normalization (zero mean, unit variance) - nnU-Net default
  - Percentile clipping (e.g., clip at 0.5th and 99.5th percentiles, then normalize) - robust to outliers
  - For fat-suppressed T1W MRI: the bright lesion signal is the key feature. Clip intensity at relevant percentiles to maximize contrast for endometriotic lesions.

### Resolution
- RSNA 2023: Resize 3D volumes to 96x256x256 (standard approach)
- RSNA 2022: 128x128 per-slice, with 128 slices
- BraTS: Crop to 128x128x128 subvolumes around ROI
- **For our task:** After liver cropping, resize to a consistent spatial resolution. The in-plane resolution of MRI is typically ~0.5-1mm; through-plane may be 3-5mm. This anisotropy favors 2.5D over 3D.

### Specific MRI Considerations for Diaphragmatic Endometriosis
- **Best sequence for detection:** Fat-suppressed T1-weighted (T1W FS) is the gold standard (Revel et al., 2016; RSNA RadioGraphics 2024). Lesions appear as hyperintense nodules.
- **Multi-sequence:** The endometriosis DL paper (2025) found best results using T2W + T1W FS pre-contrast + T1W FS post-contrast together.
- **Lesion characteristics:** Right-sided (87.5%), predominantly posterior, hyperintense on T1. Classified as micronodules (<5mm), nodules (<=3cm), or plaques (>=3cm). MRI sensitivity is 78-83% even for expert readers.
- **MRI often underestimates disease extent** compared to surgical findings.

---

## 5. Data Augmentation Strategies

### What Worked in Medical Imaging Competitions

#### Geometric Augmentations (Universally Applied)
- **Random horizontal flip** - Used in virtually every solution
- **Random affine transforms** (rotation ±15-30 degrees, scale ±10-20%, translation ±10%)
- **Random elastic deformation** - Very effective for brain tumor segmentation (σ=2 was optimal in BraTS)
- **Random cropping** - For patch-based training

#### Intensity Augmentations
- **Gaussian noise** - Modest amounts, commonly used
- **Brightness/contrast jittering** - γ correction was the 2nd most effective augmentation in BraTS benchmarks
- **Mixup (α=0.3)** - Used in RSNA 2024 4th place solution. Evidence mixed for medical imaging.

#### Medical-Imaging-Specific
- **Copy-paste augmentation of lesions** - Paste lesions from positive cases onto negative cases. Relevant for our extreme positive/negative imbalance.
- **Multi-window channel stacking** (CT-specific, not applicable to MRI)
- **Test-time augmentation (TTA)** - Horizontal flip TTA universally used. Some solutions also use multi-scale TTA.

#### What Generally Did NOT Help
- Heavy color/hue augmentation (not meaningful for grayscale medical images)
- Cutout/CutMix (can destroy small lesions)
- Excessive rotation (>45 degrees, unless anatomically justified)

### Recommendations for Our Task
1. **Safe augmentations:** horizontal flip, small rotation (±15 degrees), slight scale change (±10%), Gaussian noise
2. **Promising augmentations:** elastic deformation (σ=2), intensity γ correction
3. **Consider:** Copy-paste of lesion patches from positive to negative volumes
4. **Avoid:** Cutout/CutMix (too risky for tiny lesions), vertical flip (anatomically incorrect for abdominal MRI)

---

## 6. Optimal 2.5D Slice Stacking

### Evidence from Competitions and Papers

| Source | Slices | Channel Strategy | Notes |
|---|---|---|---|
| RSNA 2022 1st place | 5 adjacent slices | 5 channels + 1 mask = 6ch | ±2 slices from center |
| RSNA 2023 1st place | 3 slices (from 96->32x3) | 3 channels | Evenly spaced through crop |
| RSNA 2024 4th place | 3 channels | `in_channels=3` | Standard 2.5D |
| NoBrainer (ICH 2019) | 1 slice | 3 windows as channels | Single slice, 3-window |
| Systematic study (2019) | 1-25 slices tested | Channel-based | **13 slices** had slight advantage across all datasets |
| CSA-Net (2024) | Arbitrary | Cross-slice attention | Best 2.5D method in benchmarks |
| SAMBD (2022) | Multiple | Multi-branch decoder | Slice-centric attention |
| 2.5D blog (GI tract) | 3 slices | 3 channels, stride=2 | `shift(-i*stride)` with stride=2 |

### Key Findings

1. **3 channels is the pragmatic default.** It allows use of ImageNet-pretrained first-layer weights with minimal modification. Most competition winners use 3-channel input.

2. **5 channels appeared in the most technically sophisticated solution** (RSNA 2022 1st place). They used ±2 adjacent slices to form a 5-channel image for each target slice, plus a mask channel.

3. **A systematic 2019 study** (Evaluation of Multislice Inputs to CNNs for Medical Image Segmentation) found:
   - Improvement over 2D was only observed in 2/8 datasets
   - 13 input slices with a novel pseudo-3D convolution method had a slight overall advantage
   - Simply concatenating adjacent slices as channels gave marginal improvement at best
   - The benefit is task-dependent

4. **Stride/spacing matters.** For anisotropic MRI where through-plane resolution is much coarser (e.g., 3-5mm slice thickness vs. 0.5mm in-plane), adjacent slices already cover significant anatomical territory. Using stride=1 (truly adjacent) is standard.

5. **Cross-slice attention (CSA-Net, 2024)** outperformed all standard 2.5D methods in multi-task benchmarks. This is a more principled approach that explicitly models inter-slice relationships rather than relying on the backbone to implicitly learn them.

### Recommendation for Our Task

- **Start with 3 channels** (center slice + 1 above + 1 below). This allows direct use of ImageNet pretrained weights.
- **If results are promising, try 5 channels** (center ± 2 slices). Modify first conv layer: initialize center channel with pretrained weights, zero-init others.
- **Consider spacing:** If slice thickness is 3-5mm, adjacent slices already span 9-15mm. This is likely sufficient for endometriosis nodules that are typically 5mm+.
- Do NOT go beyond 7-9 channels unless you have a specific hypothesis - diminishing returns are well documented.

---

## 7. Train/Val Split Strategies

### Competition Best Practices

1. **Patient-level splits are mandatory.** Every competition enforces this. Never leak slices from the same patient across train/val.

2. **Stratified K-fold (K=4 or 5) is standard:**
   - RSNA Mammography 1st place: 4-fold patient-level splits
   - RSNA 2023 Abdomen: 4-5 fold cross-validation
   - RSNA 2024 Lumbar 2nd place: 5-fold
   - The endometriosis DL paper: 7-fold CV + 12.5% held-out test set

3. **Stratify by:**
   - Positive/negative label (critical given our imbalance)
   - Imaging site/scanner (if multi-site data)
   - Lesion size distribution (if available)

4. **With 131 positives:** 5-fold CV gives ~26 positive validation cases per fold. This is acceptable but borderline. Consider:
   - 4-fold CV (33 positives per validation fold) may be more stable
   - Stratified GroupKFold (group by patient)

5. **Full-data training for final submission** is common after CV is established, but risky without a true held-out test set.

---

## 8. Endometriosis-Specific ML Research

### Published Approaches

#### 1. Multi-Sequence MRI Classification (Springer, 2025)
- **Architecture:** 3D-DenseNet-121 classifier
- **Data:** 395 cases + 356 controls (much larger than ours)
- **Best input:** T2W + T1W FS pre-contrast + T1W FS post-contrast (3 sequences)
- **Results:** F1=0.881, AUC=0.911, sensitivity=0.976, specificity=0.720
- **Split:** 12.5% test set, 7-fold CV on remainder, patient-level
- **Key insight:** Multi-sequence input significantly outperformed single-sequence

#### 2. AI-Based MRI Reading Support (AMP) (Nature Scientific Reports, 2025)
- **Architecture:** nnU-Net for plaque segmentation + nnU-Net for ovarian endometriotic cysts + LightGBM for adhesion detection
- **Performance:** Mean Dice 0.293 for plaque segmentation (!) and 0.580 for OEC segmentation
- **Key insight:** Plaque segmentation is very hard even for nnU-Net. Recall improved from 0.73 to 0.91 with AI assistance for radiologists.
- **Notable:** This used 3D multi-slice MRI processing

#### 3. Pelvic Organ Segmentation for Endometriosis (Nature Scientific Data, 2025)
- **Architecture:** nnU-Net (baseline) vs. RAovSeg (ResNet classifier + Attention U-Net)
- **Data:** 51 subjects multicenter + 81 subjects single-center
- **Custom network beat nnU-Net** on this specific task
- **Loss:** Tversky loss with α=0.8, β=0.2, γ=1.33

### Diaphragmatic Endometriosis Imaging Characteristics

Based on radiology literature (Revel et al. 2016, RSNA RadioGraphics 2024, multiple reviews):

- **Location:** Right-sided in 87.5% of cases, predominantly posterior
- **Morphology:** Nodules (72%), micronodule clusters (<5mm), plaques (>=3cm), or focal liver herniation
- **Signal:** Hyperintense on fat-suppressed T1W (blood products) and T2W. **Best depicted on fat-suppressed T1W sequences.**
- **Fat-suppressed T1W GRE sequence** enhances detection of small lesions with high sensitivity
- **MRI sensitivity:** 78-83% for expert readers; MRI misses small superficial nodules and diaphragmatic holes
- **MRI underestimates disease extent** compared to surgical findings
- **MRI during menses** may improve detection (cyclic hemorrhagic activity)
- **Preoperative MRI diagnosis was established in only 2/19 patients** in one surgical series, highlighting the difficulty

---

## 9. nnU-Net and MONAI Approaches

### nnU-Net
- Self-configuring: automatically determines preprocessing, architecture, training parameters
- Default loss: DiceCE
- Works well for most segmentation tasks but **struggles with very small lesions** (multiple GitHub issues document this)
- **Known limitation:** Standard Dice loss means small lesions contribute negligibly to the gradient
- **Solutions within nnU-Net:**
  - Custom loss functions (blob loss, CC-DiceCE, CATMIL) - all available as plugins
  - Adjust patch size to match lesion scale
  - Foreground oversampling (already built-in: 33% foreground patches by default)
  - Consider the residual encoder variant for better feature extraction

### MONAI
- Key transforms for our task:
  - `RandCropByPosNegLabeld` - balanced pos/neg patch sampling
  - `ScaleIntensityRanged` - clip and normalize intensity
  - `CropForegroundd` - crop to body region
  - `RandAffined` - geometric augmentation
  - `SlidingWindowInferer` - for inference on full volumes
- Standard approach: 3D UNet or SegResNet with DiceLoss + sliding window inference
- For small datasets: pretrained weights from MedicalNet are available

---

## 10. Pitfalls and Concerns

### Data-Specific Concerns
1. **131 positives is very small** for deep learning segmentation. Expect high variance across folds. Prioritize strong regularization (augmentation, weight decay, dropout, early stopping).
2. **~1000 negatives are a resource, not just noise.** Use them for pretraining or representation learning, or as hard negatives during training.
3. **Annotation quality is critical** at this sample size. Verify mask quality thoroughly during EDA.
4. **Multi-site variability:** If data comes from multiple scanners/sites, intensity distributions will vary. Z-score normalization or histogram matching may be needed.

### Methodological Pitfalls
5. **Do not use CutMix/CutOut** - these can destroy the tiny lesions you're trying to detect.
6. **Be careful with aggressive downsampling** - at 5mm slice thickness and <5mm lesion size, lesions may span only 1-2 slices. Downsampling in z can lose them entirely.
7. **Evaluate with instance-level metrics** (lesion-wise F1, sensitivity) not just Dice. A model can get decent Dice by finding large lesions and missing small ones.
8. **MRI anisotropy:** Through-plane resolution is typically much worse than in-plane. Don't treat voxels as isotropic.
9. **Beware of trivial solutions:** A classifier that predicts "no lesion" for everything achieves ~88% accuracy. Use proper metrics (F1, AUC, sensitivity at fixed specificity).

### Practical Concerns
10. **Overfitting risk is high.** The endometriosis plaque segmentation paper achieved only Dice=0.293 with nnU-Net on a similar task. Temper expectations.
11. **The 2.5D approach may be sensitive to slice selection.** If lesions span only 1-2 slices, the center slice matters a lot. Consider training on all slices in the ROI, not just the "best" ones.
12. **Ensembling is almost always worth it** in competitions. Even 2-3 models with different seeds or architectures provide meaningful improvement.

---

## 11. Recommended Experimental Plan

Based on all the above evidence, here is a prioritized plan:

### Phase 1: Baseline
1. **Preprocessing:** Crop volumes around the right hemidiaphragm/liver region using either (a) atlas-based coordinates, (b) a pretrained liver segmentation model, or (c) manual coordinates from EDA
2. **Architecture:** 2.5D approach: EfficientNetV2-S or ConvNeXt-tiny backbone, 3-channel input (center slice ± 1), pretrained on ImageNet
3. **Loss:** DiceCE or Dice + Focal Loss
4. **Split:** 4-fold stratified GroupKFold (patient-level)
5. **Augmentation:** horizontal flip, rotation ±15 degrees, scale ±10%, Gaussian noise

### Phase 2: Improvements
6. Try 5-channel input (center ± 2 slices)
7. Add segmentation mask as auxiliary channel (if available from stage-1 coarse segmentation)
8. Add blob loss or CC-DiceCE for better small lesion recall
9. Try copy-paste augmentation of lesion patches
10. Ensemble 2-3 models (different seeds and/or backbones)

### Phase 3: Advanced
11. Two-stage: slice-level feature extraction -> sequence model (GRU/LSTM) for volume-level prediction
12. nnU-Net with custom loss as a fully-3D comparison baseline
13. Multi-sequence input if multiple MRI sequences are available

---

## References

### Competition Solutions
- RSNA 2019 ICH 1st place: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
- RSNA 2019 ICH 2nd place: https://github.com/darraghdog/rsna
- RSNA 2022 C-Spine 3rd place: https://github.com/darraghdog/RSNA22
- RSNA 2022 C-Spine 5th place: https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place
- RSNA 2022 C-Spine 6th place: https://github.com/i-pan/kaggle-rsna-cspine
- RSNA 2023 Abdomen 1st place: https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- RSNA 2023 Abdomen 2nd place: https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
- RSNA 2024 Lumbar 2nd place: https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification
- RSNA 2024 Lumbar 4th place: https://github.com/yu4u/kaggle-rsna2024-4th
- RSNA Mammography 1st place: https://github.com/dangnh0611/kaggle_rsna_breast_cancer

### Loss Functions
- Blob Loss: https://arxiv.org/abs/2205.08209
- CC-DiceCE: https://github.com/TIO-IKIM/Learning-to-Look-Closer (arXiv:2511.17146)
- CATMIL: https://github.com/luumsk/SmallLesionMRI (arXiv:2604.08015)
- Unified Focal Loss: https://github.com/mlyg/unified-focal-loss (arXiv:2102.04525)
- Focal Tversky Loss: arXiv:1810.07842
- Inverse Loss Reweighting: https://arxiv.org/pdf/2007.10033

### 2.5D Methods
- Systematic evaluation of multislice inputs: arXiv:1912.09287
- CSA-Net (Cross-Slice Attention): https://github.com/mirthAI/CSA-Net
- 2.5D SAMBD: arXiv:2203.03640
- 2.5D Blog tutorial: https://awsaf49.github.io/blog/2023/2.5d-training/

### Endometriosis
- Multi-sequence MRI DL for endometriosis: https://link.springer.com/article/10.1007/s00261-025-04942-8
- AMP reading support: https://www.nature.com/articles/s41598-025-30277-x
- Pelvic segmentation dataset: https://www.nature.com/articles/s41597-025-05623-3
- MR diagnosis of diaphragmatic endometriosis: https://link.springer.com/article/10.1007/s00330-016-4226-5
- Diaphragmatic endometriosis review: https://pmc.ncbi.nlm.nih.gov/articles/PMC11604425/
- RadioGraphics endometriosis of the diaphragm: https://pubs.rsna.org/doi/full/10.1148/rg.240153

### Frameworks
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- MONAI tutorials: https://github.com/Project-MONAI/tutorials
- nnDetection: https://github.com/MIC-DKFZ/nnDetection
