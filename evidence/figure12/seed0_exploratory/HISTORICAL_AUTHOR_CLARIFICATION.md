# Authors' Clarification Of The Original Evaluation

## Source And Scope

This is an author-supplied clarification of the original Figure 12 procedure,
recorded on 2026-11-27. It is distinguished from facts independently established
by the retained logs, code and prediction records.

According to the authors' clarification:

1. Images were selected separately for each dataset to form a subset of
   approximately 50 images, depending on the availability of suitable images
   in each class. Fifty was an approximate target, not an exact common count.
2. The historical Modest-Align values were image-to-text top-1 (R@1) results
   from retrieval evaluation code, averaged over multiple seeds. The historical
   CLIP values came from one evaluation run of a separate classification script.

The clarification does not specify the exact number of historical seeds or
the per-run image lists. Whether images were redrawn under each seed is not
asserted. The released records do not independently reconstruct every original
percentage. This clarification is not presented as a recovered execution log.

## Interpretation Of The Arithmetic Checks

The `3 x 50` column in `CALCULATIONS.md` tests exactly three equal-weight
evaluations with exactly 50 image queries in each. Its False results reject
that specific assumption; they do not test every possible multi-seed protocol.
Neither exactly three seeds nor exactly 50 images per run is established by
the authors' clarification. The tests are retained unchanged, not relabeled
as reproductions of the historical experiment.

## Separate New Evaluation

The revised figure uses a documented new, fixed-list, class-level evaluation.
Its illustrated COCO-566K adapter is seed 0, selected after inspection of five
checkpoints. It is not the historical seed average. The lists contain
45/47/57/49/63 images, and both methods use the same images and prompts.
Historical values, the new five-checkpoint results and the new seed-0 plot
remain separate records.
