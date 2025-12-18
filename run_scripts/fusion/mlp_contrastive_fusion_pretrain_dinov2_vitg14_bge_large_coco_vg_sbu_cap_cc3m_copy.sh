# CUDA_VISIBLE_DEVICES=1 python train.py --cfg-path lavis/projects/fusion/mlp_contrastive_fusion_pretrain_dinov2_vitg14_bge_large_coco_vg_sbu_cap_cc3m.yaml
#!/bin/bash
# Lock memory
# sudo python -c "import ctypes; ctypes.CDLL('libc.so.6').mlockall(1)"

# Set CUDA device and run Python training script
CUDA_VISIBLE_DEVICES=7 python train.py --cfg-path lavis/projects/fusion/mlp_contrastive_fusion_pretrain_dinov2_vitg14_bge_large_coco_vg_sbu_cap_cc3m.yaml
