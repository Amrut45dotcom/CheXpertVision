# CheXpertVision (WIP)

Chest X-ray analysis system combining computer vision and LLMs.

## Current Progress

### Dataset
* CheXpert small dataset
* 5 target classes: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
* 50% of patients used for faster iteration
* Patient-level 70/15/15 train/val/test split

### Label Preprocessing
* NaN → 0 (absence)
* Uncertain (-1) → 1 (U-Ones strategy)

Rationale: prioritize sensitivity in medical context, minimize false negatives

### Model
* EfficientNet-B0, ImageNet pretrained
* Final layer: `nn.Linear(→5)`
* Loss: `BCEWithLogitsLoss` with `pos_weight` (computed from train set only)
* Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
* AMP (mixed precision) + cudnn.benchmark

### Training Setup
* Batch size: 64
* Hardware: RTX 2050, 4GB VRAM
* Normalization: mean=0.5330, std=0.0349 (computed from train set)

### Augmentation (Train only)
* RandomRotation(±10°)
* RandomHorizontalFlip(p=0.5)
* ColorJitter(brightness=0.2, contrast=0.2)
* RandomResizedCrop(224, scale=(0.85, 1.0))

### Results (EfficientNet-B0, 8 epochs)

| Class | AUC |
|-------|-----|
| Atelectasis | 0.6966 |
| Cardiomegaly | 0.8167 |
| Consolidation | 0.6530 |
| Edema | 0.8369 |
| Pleural Effusion | 0.8600 |
| **Mean AUC** | **0.7726** |

**Observations:**
* Val loss diverges from train loss at epoch 4 — overfitting onset
* Augmentation slows divergence but does not eliminate it
* AUC plateaus around 0.77

## Roadmap

### Week 1 (Current)
- [x] Data pipeline
- [x] Baseline EfficientNet-B0
- [x] pos_weight, AMP, AdamW
- [x] Augmentation
- [ ] Architecture comparison (ResNet50, DenseNet121)

### Week 2
- [ ] Grad-CAM visualization
- [ ] MC Dropout uncertainty estimation
- [ ] Hyperparameter tuning (lr scheduler, dropout)
- [ ] LLM integration

## Status
Work in progress — architecture comparison next.