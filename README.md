# Modest-Align

This is the official implementation of the paper: **Modest-Align**.

## Introduction
We propose a novel multimodal fusion method that leverages multi-distribution noise mixing to enhance robustness and alignment. Our code is based on the [LAVIS](https://github.com/salesforce/LAVIS) library.

<img width="839" height="284" alt="image" src="https://github.com/user-attachments/assets/ce31aaca-a4c6-4e73-89eb-bc5484326c9c" />

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n modest-align python=3.8
conda activate modest-align
```

2. Build from source

```bash
git clone https://github.com/your-repo/modest-align
cd modest-align
pip install -e .
```

## Getting Started

### Dataset Preparation
Please refer to `lavis/datasets/download_scripts` for scripts to download the required datasets.

### Training

To train the model with multi-distribution noise mixing:

```bash
bash run_scripts/fusion/mlp_contrastive_fusion_pretrain_dinov2_vitg14_bge_large_coco_vg_sbu_cap_cc3m.sh
```

### Evaluation

To evaluate the model:

```bash
bash run_scripts/fusion/mlp_contrastive_fusion_retrieval_dinov2_vitg14_bge_large_coco.sh
```

## Citation
If you find this work useful in your research, please cite the following paper:
```
@article{liu2025modest,
  title={Modest-Align: Data-Efficient Alignment for Vision-Language Models},
  author={Liu, Jiaxiang and Wang, Yuan and Du, Jiawei and Zhou, Joey Tianyi and Xu, Mingkun and Liu, Zuozhu},
  journal={arXiv preprint arXiv:2510.21606},
  year={2025}
}
```

## Acknowledgement
Our code is developed based on [LAVIS](https://github.com/salesforce/LAVIS). We thank the authors for their great work.
