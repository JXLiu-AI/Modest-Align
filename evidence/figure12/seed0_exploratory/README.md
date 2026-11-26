---
library_name: pytorch
pipeline_tag: zero-shot-image-classification
tags:
  - modest-align
  - frozen-features
  - research
---

# Modest-Align: Exploratory COCO Seed-0 Classification

This release identifies the single COCO-trained RP+ES adapter checkpoint used
in the revised supplementary classification figure. It contains the inference
architecture, documented score-to-metric calculations, and all
five training-seed results for context. Frozen DINOv2, BGE and CLIP weights are
not included.

## Scope And Selection

The displayed checkpoint is training seed **0**, selected training epoch **202**
by the original Flickr30K retrieval criterion, from training completed over
500 epochs on **566K COCO pairs**. This is not the 3.5M-pair main-result setting.
The decision to illustrate seed 0 was made after exploratory inspection of five
checkpoints. It is not a prespecified estimate or the five-seed mean; see
`all_five_seeds.csv`. No general medical classification advantage is claimed.

This is a documented new class-level evaluation, not a reconstruction of the
original Figure 12 percentages. The original run-to-number aggregation is not
established by the retained records. No historical resampling or multi-seed
procedure is asserted here.

## Results

All values are percentages. Balanced accuracy is mean per-class recall.

| Dataset | Images | CLIP top-1 | Adapter top-1 | CLIP balanced | Adapter balanced |
|---|---:|---:|---:|---:|---:|
| CheXpert | 45 | 55.56 | 60.00 | 49.07 | 52.78 |
| Shenzhen | 47 | 74.47 | 74.47 | 50.00 | 50.00 |
| BrainTumor | 57 | 71.93 | 71.93 | 50.00 | 50.00 |
| IDRiD | 49 | 59.18 | 57.14 | 20.00 | 19.31 |
| Dogs & Cats | 63 | 100.00 | 100.00 | 100.00 | 100.00 |

The four medical subsets are small and class-imbalanced. Their apparently high
top-1 values can reflect majority-class predictions: for example, both models
predict the majority class for every Shenzhen and BrainTumor image. These
subsets are not official clinical test splits or verified patient-level splits.

## Recompute The Published Arithmetic

```bash
python verify_scores.py
python -m unittest -v test_verify_scores.py
```

These commands use only Python's standard library and recompute all 522 released
image decisions. `predictions.jsonl` uses dataset-local public IDs, not image
paths or patient identifiers.

## Load The Adapter

```python
import torch
from adapter_model import load_adapter

model = load_adapter("model.safetensors")
with torch.inference_mode():
    scores = model(image_features, class_text_features)
    predicted_class = scores.argmax(dim=1)
```

Inputs are unnormalized frozen DINOv2 ViT-g/14 image features of shape
`[N,1536]` and BGE-large-en-v1.5 class-text CLS features of shape `[C,1024]`.
The adapters project them to 512 dimensions, normalize, and compute cosine
similarity. No training, test-time perturbation, mixing or dropout is applied.
For an authorized feature file:

```bash
python -m pip install -r requirements.txt
python verify_weights.py
python infer_features.py --weights model.safetensors --features authorized_features.npz --output results.json
```

The NPZ contains `image_features`, `text_features`, and optional integer
`targets`. See `evaluation_protocol.json` for exact preprocessing, class counts,
prompts and limitations. `extract_features.py` can produce this NPZ from
authorized local images and separately obtained encoders.

## Data Access And Verification Limits

Raw medical images, medical image embeddings, private file-path mappings,
image-content hashes and private correspondence are not distributed. Exact
pixel-level reproduction of the small subsets therefore requires authorized
source images and the original private mapping; the public score arithmetic
alone is not a pixel-level reproduction. Once the separately hosted weights are
available, readers can test the adapter on their own authorized images/features. Obtain the appropriate source
data permissions separately.

When distributed, the Safetensors file contains only the adapter tensor state, not optimizer
state or executable pickle content. `config.json` identifies the source
checkpoint by SHA-256; `release_manifest.json` verifies the distributed files.
Weight hosting is recorded separately in `distribution_status.json`; code and
score publication alone does not imply the weights are already hosted.
The frozen encoders and their terms are separate from this release. No new
license grant for third-party encoders or datasets is implied.
