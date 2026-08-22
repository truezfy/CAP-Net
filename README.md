# CAP-Net: CT-Assisted PET Network for Efficient 3D Tumor Segmentation

**Information Fusion, 2026** | [DOI: 10.1016/j.inffus.2026.104690](https://doi.org/10.1016/j.inffus.2026.104690)

## Overview

CAP-Net is a novel CT-assisted PET network for efficient 3D tumor segmentation in PET/CT images.
![CAP-net](figure/net_final_01.png)

## Abstract

Fusing complementary information from PET and CT is beneficial for tumor segmentation. However, few studies have considered the asymmetric roles of the two modalities: PET directly reflects tumor uptake and enables explicit tumor localization, whereas CT primarily provides anatomical context. Ignoring this modality asymmetry not only increases parameter requirements for PET/CT feature fusion, but also risks diluting discriminative PET tumor signals with abundant CT background, which is particularly detrimental for small tumors. To address these issues, we propose a novel CT-assisted PET network (CAP-Net) for efficient 3D tumor segmentation. CAP-Net integrates local features extracted by a shallow CNN branch with asymmetric feature fusion (SCAF) and global features obtained by the CAP-RWKV branch. Both branches treat PET as the dominant modality. CAP-RWKV incorporates a Local Variance Similarity-based Token Area Allocation (LVS-TAA) strategy, assigning key tokens to high-uptake PET regions while merging background into larger tokens, maximizing prior information retention. In addition, we introduce a Key Token Cross-Entropy (KT-CE) loss to mitigate class imbalance. Experiments on three public datasets demonstrate that CAP-Net achieves state-of-the-art performance, significantly improving segmentation accuracy while reducing parameters by 57.4% compared to the latest efficient model(Mobile U-ViT). 

## Requirements

Python 3.9.23, PyTorch 2.8.0, CUDA 12.8 (for GPU support).

Key dependencies: torchio, nibabel, SimpleITK, numpy, scipy, scikit-learn, matplotlib, tqdm, PyYAML.

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@article{ZHANG2027104690,
title = {Rethinking modality fusion: CT-assisted PET network for efficient 3D tumor segmentation},
journal = {Information Fusion},
volume = {138},
pages = {104690},
year = {2027},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2026.104690},
url = {https://www.sciencedirect.com/science/article/pii/S156625352600566X}
}
```
