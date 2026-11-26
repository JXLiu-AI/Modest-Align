# Historical Experiment Evidence

This record preserves the surviving project material separately from the new
matched classification evaluation. Historical log files are retained verbatim,
with their source hashes, rather than generated from the revised figure.

## Retained Records

- [Historical reported values](historical_reported_values.csv) preserve every
  earlier plotted percentage, including unfavorable comparisons.
- [Original metric logs](historical_records/retained_logs/) contain 36 retained
  metric rows across 17 files. Each file is byte-identical to its archived source.
- [Source excerpts](historical_records/source_excerpts/) retain the relevant
  CLIP classification, retrieval scoring, similarity-matrix and dataset-mapping
  functions. They are exact line excerpts, not rewritten implementations.
- [Provenance](historical_records/provenance.json) gives original-file SHA-256
  hashes, source IDs, source line spans and recorded modification timestamps.
  Timestamps are filesystem metadata, not an independent proof of execution.
- [Metric ledger](legacy_metric_rows.csv) and
  [arithmetic checks](CALCULATIONS.md) show how the retained log fields relate
  to their stored summaries.

The logs and source code document genuine surviving experimental artifacts.
They do not by themselves identify the exact image groups, checkpoint and
aggregation that produced every historical Figure 12 percentage. A matching
number from a different task is not treated as evidence for that figure.
No undocumented historical seed-averaging or resampling procedure is asserted.

## New Documented Evaluation

The revised illustrative figure is separately documented in
[seed0_exploratory](seed0_exploratory/README.md). Its fixed lists, class prompts,
per-image cosine scores, predictions, correct counts and all five seed results
are supplied. Historical percentages and newly recalculated percentages are
never merged or relabeled as the same experiment.

No private correspondence, raw medical images, medical image embeddings,
patient identifiers or private server paths are published here.
