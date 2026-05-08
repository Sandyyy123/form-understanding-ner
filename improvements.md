# Improvements - Project 18 (FUNSD Form Understanding)

Role B (IMPROVER) review of the Phase 1 scaffold. Recommendations only; no files modified.

---

## Top recommendation

**Replace the placeholder linking ranker in `src/model_advanced.py` with a properly trained `LinkingHead` that consumes LayoutLMv3 hidden states, and report directed-edge linking F1 (not undirected) so the scaffold can actually generate Table 2 instead of `<TBD>`.** Linking is the headline IDP differentiator over plain entity tagging, and the current spatial-proximity heuristic is a known lower bound (the manuscript itself says so in 2.4 and 4.5). Concretely: take the last-layer token embeddings from the trained LayoutLMv3, average-pool per entity using `pool_entity_embeddings()` (already defined but unused), build positive pairs from `linking_pairs` and 5:1 negatives across all (question, any-other-entity) combinations, train the `LinkingHead` MLP for 3-5 epochs with AdamW 1e-4, and score on the test split using both directed and undirected matching. This is a 2-3 hour code change that converts the linking story from "placeholder" to "headline result" and unlocks the manuscript's strongest single contribution.

---

## Weakness 1 - Linking head is a heuristic, not the trained model the manuscript advertises (HIGH)

The advanced script defines `LinkingHead` but the `train_and_eval_linking_head()` function falls back to nearest-centroid matching. The manuscript text in 2.4 promises a trained head and section 3.2 reserves a row for it. Implement the torch training loop sketched as a TODO: run a forward pass on each train form, pool entity embeddings via `pool_entity_embeddings()`, sample 5:1 negative-to-positive pairs, optimise binary cross-entropy on the head's two-logit output, and freeze the LayoutLMv3 backbone for the first epoch then unfreeze with discriminative LRs (1e-5 backbone, 1e-4 head). Without this, the Linking row in Table 2 has no real number.

## Weakness 2 - No multi-seed averaging despite small training set (HIGH)

149 forms is exactly the regime where seed variance is 1-2 absolute F1 points (the manuscript itself flags this in 4.2 and limitation 1). The scaffold pins seed 42 once. Add a `--seeds 42,1337,2024` CLI flag to both scripts, run the full pipeline three times, write `metrics_baseline_seed{N}.json` per run, and report mean ± std in Table 1. This is a 30-line wrapper change and produces the only credibility signal that lets a client trust the macro-F1 gap is real and not noise.

## Weakness 3 - Class imbalance is acknowledged but never addressed (MEDIUM)

Section 4.4 says "if the macro-F1 gap between O and B-header exceeds 0.4, switch to Focal Loss" but neither script implements that fallback. Add a `loss_type=focal|ce` config and a `FocalLoss` module (gamma=2, alpha computed from inverse class frequency on the train split). On FUNSD the `B-header` class has under 200 instances in train versus 7000+ for `O`, so this is a textbook focal-loss case. Run both losses, report both rows, let the data decide.

## Weakness 4 - No requirements.txt, no environment pinning (HIGH)

The manuscript section 2.6 claims "the team's `requirements.txt` is version-pinned" but no such file exists in the project root. Without pinned versions of `transformers`, `tokenizers`, `torch`, `seqeval`, the LayoutLMv3 numbers are not reproducible: HuggingFace has shipped breaking changes to the LayoutLMv3 processor between 4.30 and 4.40, and `seqeval` had a metric-rounding fix in 1.2.2. Add `requirements.txt` listing exactly: `transformers==4.44.0 torch==2.4.0 tokenizers==0.19.1 sklearn==1.5.0 Pillow==10.4.0 seqeval==1.2.2 matplotlib==3.9.0 seaborn==0.13.2 numpy==1.26.4`.

## Weakness 5 - Baseline ignores layout entirely; a stronger text-only competitor is trivial to add (MEDIUM)

Limitation 2 says "a more competitive text-only baseline would inject reading-order signals via spatial sorting before tokenization." This is half a day of work: sort words by `y0 // 30` (row band) then `x0` before flattening, and re-run the BERT fine-tune. On SROIE this trick alone closes ~30% of the gap to LayoutLM and is the fairest text-only competitor. Add it as `model_baseline_v2.py` or as a `--reading-order spatial|file-order` flag. Without it, the text-only baseline is artificially weak and the LayoutLMv3 gap is overstated.

## Weakness 6 - No calibration or confidence reporting (MEDIUM)

IDP production deployments are routed by confidence: high-confidence predictions go straight-through, low-confidence get human review. The scaffold reports F1 only and not expected calibration error (ECE), reliability diagrams, or threshold-routed precision-recall curves. Add `sklearn.calibration.calibration_curve` over the softmax probabilities of the top-1 BIO tag, compute ECE in 10 bins, and plot a reliability diagram next to the confusion matrix. This is the metric that actually maps to the EUR 2-6 per-page cost story in the brief.

## Weakness 7 - Linking PR curve is built from wrong score distribution (MEDIUM)

In `train_and_eval_linking_head()`, `y_true` and `y_score` are appended only inside the inner answer loop, so they capture every (q, a) pair as a separate scoring decision but the final pred_pairs uses argmax. The PR curve and the F1 number then describe two different decision rules. Either build the PR curve over the same argmax decisions (one prediction per question), or report sweep-threshold linking F1 as a separate metric. The current mix-up will produce a misleading PR curve in Figure 3.

## Weakness 8 - No fairness or robustness audit on the demographic / OCR-noise dimension (LOW)

FUNSD forms span business domains (medical, legal, government) and the linking task is sensitive to layout density. A 2-page audit slicing the test set by form_id pattern (forms originally from the FDA tobacco-litigation subset versus the rest) and reporting per-slice F1 would give the manuscript a fairness-style robustness section. Also add a Gaussian-noise + 90/180-degree rotation OCR-noise simulation on test images (cheap to script with PIL) and report degradation, since the brief mentions OCR robustness as a production concern.

---

## Priority summary

| # | Weakness | Priority |
|---|---|---|
| 1 | LinkingHead never actually trained | HIGH |
| 2 | Single seed, no variance reporting | HIGH |
| 4 | No pinned requirements.txt | HIGH |
| 3 | Focal Loss promised but not implemented | MEDIUM |
| 5 | Text-only baseline ignores spatial reading order | MEDIUM |
| 6 | No calibration / ECE reporting | MEDIUM |
| 7 | Linking PR curve and F1 use inconsistent decision rules | MEDIUM |
| 8 | No slice-level robustness or OCR-noise audit | LOW |

Implementing items 1, 2, and 4 alone would convert this scaffold from a code-only Phase 1 deliverable into a fully reproducible, defensible benchmark report.
