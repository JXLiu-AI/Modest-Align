# Evaluation Protocol and Computation Definitions

## 1. Fixed Convenience Subsets

All entries of the recovered lists were used in the new evaluations.
Their original selection process is unknown; these are not claimed to be
official, random or patient-independent test splits.

| Dataset | N | Class counts |
| --- | ---: | --- |
| CheXpert | 45 | Abnormal 27; normal 18 |
| Shenzhen | 47 | Tuberculosis 35; normal 12 |
| BrainTumor | 57 | Glioblastoma 41; primary CNS lymphoma 16 |
| IDRiD | 49 | No apparent 4; mild 7; moderate 5; severe 4; proliferative 29 |
| Dogs & Cats | 63 | Cat 31; dog 32 |

The current subset sizes are not established as the denominators for all
historical figure values. Alternative historical manifests also survive.
These are recovered categorical tasks, not a complete CheXpert multilabel
benchmark or a general clinical validation.

## 2. New Class-Level Inference

CLIP uses the fixed OpenAI ViT-B/32 model, its standard image transform and
tokenizer, and one class-name prompt per class. The adapter evaluations use
frozen DINOv2 ViT-g/14 and BGE-large-en-v1.5 encoders. DINOv2 images are
resized and center-cropped to 224 x 224 and normalized with ImageNet channel
statistics. BGE uses its unnormalized CLS output after the original
lowercase/punctuation caption preprocessing.

The full prompt strings and class order are present on every score row.
Each class occurs exactly once in that row's candidate list. No class is
selected or excluded by performance. All models are in evaluation mode,
without test-time perturbation, feature mixing, dropout, prompt tuning or
classification fitting. Inference used CPU fp32. Historical GPU precision
equivalence is not asserted.

The image/text features are projected through the respective adapters and
then L2-normalized. Their dot products are cosine class scores:

```text
v = image_adapter(frozen_image_feature)
t[c] = text_adapter(frozen_text_feature(class_prompt[c]))
score[c] = dot(v / norm(v), t[c] / norm(t[c]))
prediction = first argmax(score)
```

CLIP applies the corresponding cosine rule to its own image/text features.
The target class is used only for scoring correctness, not for prediction.
Every released `cosine_scores` row is an input to the independent calculator.
The released arithmetic begins at these scores, not at private images.

## 3. Distinct Adapter Regimes

**COCO_RP_ES:** all five existing controlled full-COCO checkpoints, trained
with 566K pairs for 500 epochs and selected by the original Flickr30K
retrieval criterion. Seeds are exactly 0, 1, 3, 7 and 42. The underlying
controlled training uses RP scale 0.01 and ES 0.1. Selected epochs and all
checkpoint hashes are in `protocol.json`. No classification-result-based
checkpoint or seed selection was performed.

**candidate_20240906195:** a separate retrieval checkpoint, selected epoch
241, seed 42. Its archived copy has the same whole-file hash. Its saved
configuration records COCO/VG/SBU/CC3M training, but not all hard-coded RP/ES
settings. The original adapter class loads it strictly. Its classification
prototype initializer is explicitly reconstructed; the original historical
initializer and Figure-12 prediction linkage remain unverified. Do not
present this regime as a fully identified historical Modest-Align run.

## 4. Classification Metrics and Seed Aggregation

For each cell:

```text
correct_i = 1 if predicted_class_i == target_class_i else 0
top1_percent = 100 * sum(correct_i) / N
recall_c_percent = 100 * correctly_predicted_class_c / samples_in_class_c
balanced_accuracy_percent = sum(recall_c_percent) / number_of_classes
```

For the five COCO adapters, compute each metric separately per seed, then:

```text
mean = sum(seed_values) / 5
sample_sd = sqrt(sum((seed_value - mean)^2) / 4)
```

This is not three repetitions of a 50-image test. No patient-level
confidence interval or significance test is inferred from this seed SD.

## 5. Surviving Historical Retrieval Metrics

The retained records contain:

```text
txt_r_mean = (txt_r1 + txt_r5 + txt_r10) / 3
img_r_mean = (img_r1 + img_r5 + img_r10) / 3
r_mean = (txt_r_mean + img_r_mean) / 2
agg_metrics = txt_r_mean
```

Pair-index positives are not automatically the same as semantic class
positives when class descriptions repeat. A percentage alone does not
identify its metric or denominator. The original output naming also puts
multiple invocations within the same ten-minute directory bucket; a bucket
is not a unique independent run or dataset label.

The ledger supplies all surviving rows without selecting favorable ones.
It does not supply the missing final aggregation that produced the five
historical percentages. No such formula is invented here.

## 6. Text-Cache Identity Test

One surviving cache contains 48 text vectors and two mappings, paired with
three visual features. Its text vectors match separately encoded characters
from three copies of `This is a NORMAL`, not three whole-caption vectors.
The raw retrieval loader expects a list of captions; a bare string can be
iterated character by character before tokenization. This is distinct from
ordinary subword tokenization of an intact caption.

The generic text vectors, fixed identity-test prompt, comparison tolerances
and every resulting cosine/difference are released. This observation is
not attributed to every historical figure value. Original files were not
modified to obtain the match. No combinations were searched to fit a
published percentage.

## 7. Limits of the Claims

The new evaluations show dataset-dependent results, not a general medical
zero-shot advantage. Several medical subsets have near-chance balanced
accuracy or single-class prediction behavior. The arithmetic can be
reproduced completely from released scores, but input sampling provenance,
historical grouping and pixel-level reruns require additional retained data.
