# Figure 12 Evidence Index

## 1. Historical Experiment Provenance

Read [HISTORICAL_EVIDENCE.md](HISTORICAL_EVIDENCE.md) for retained original
metric logs, exact code excerpts and source hashes. This documents what
survives from the earlier project without inventing a missing run-to-number
aggregation. The authors' subsequent description of approximately 50 images
per dataset, a multi-seed Modest-Align mean and a single CLIP evaluation is
separately attributed in
[HISTORICAL_AUTHOR_CLARIFICATION.md](HISTORICAL_AUTHOR_CLARIFICATION.md).
It is not presented as an independently recovered historical run ledger.

## 2. New Evaluation And Complete Arithmetic

Read the [seed-0 evaluation record](seed0_exploratory/README.md).
For each image, select the class with the largest stored cosine similarity,
compare it with the recorded label, count correct predictions and divide by
the dataset's image count. Balanced accuracy averages the per-class recalls.

| Dataset | Images | CLIP correct | CLIP top-1 (%) | Modest-Align correct | Modest-Align top-1 (%) |
|---|---:|---:|---:|---:|---:|
| CheXpert | 45 | 25 | 55.56 | 27 | 60.00 |
| Shenzhen | 47 | 35 | 74.47 | 35 | 74.47 |
| BrainTumor | 57 | 41 | 71.93 | 41 | 71.93 |
| IDRiD | 49 | 29 | 59.18 | 28 | 57.14 |
| Dogs & Cats | 63 | 63 | 100.00 | 63 | 100.00 |

Run `python verify_scores.py` inside `seed0_exploratory` to check every one
of the 522 recorded decisions, then run
`python -m unittest -v test_verify_scores.py` for the six validation tests.
The displayed seed 0 was selected after inspecting five checkpoints; all
five results remain available and the plot is not a prespecified seed average.
This COCO 566K-pair checkpoint is distinct from the 3.5M-pair main setting.

Arithmetic reproduction uses the released scores. Independent image-to-score
inference additionally requires the corresponding authorized images and frozen
encoders; protected medical images and private file mappings are not included.

## 3. Verified Adapter Weights

The tensor-only COCO seed-0 adapter is publicly available on
[Hugging Face](https://huggingface.co/AlexLiu2023/Modest-Align-COCO-seed0/tree/7903592a3b9d44614eea90e7ac62ea979f4a14f0).
The public Hub file size and LFS SHA-256 match the audited local export:
441,767,204 bytes, SHA-256
`6df22bca7bb1a9af2b50844e395537a1d73f117a9e223477d4caffd3e91432a5`.
All 57 tensors and a deterministic synthetic inference fixture were checked
before upload. Frozen encoder weights and medical images are not included.
