# Validation Report - Project 18 (FUNSD Form Understanding)

## Overall verdict: PASS-WITH-WARNINGS

Compact summary: All structural checks pass. Notebook JSON parses, both Python scripts have valid syntax, manuscript is 4,237 words (inside the 4000-5000 target band), IMRaD sections are complete, no em-dashes anywhere, no AI-tell phrases, checkpoint schema includes all four required keys. Five live CrossRef checks all returned HTTP 200; titles match for FUNSD, LayoutLMv3, LiLT, ResNet, and the BERT DOI resolves but its CrossRef metadata has an empty title field (downstream metadata-quality WARN, not a fabrication). Two real defects: the deliverables HTML carries one external href (the FUNSD project URL), violating the inline-only rule, and the manuscript contains one orphan citation `[Xu 2022]` on line 94 (LayoutXLM is filed in references.md as Xu 2021, not 2022). No saved-model artefacts in `deliverables/` (scaffold-only project, expected). No FAIL findings; both warnings are quick fixes.

---

## Task 1. Notebook validity

- [PASS] `notebooks/01_EDA.ipynb` parses as JSON via `python3 -c "import json; json.load(open(...))"`.

## Task 2. Python script syntax

- [PASS] `src/model_baseline.py` parses via `ast.parse` (no SyntaxError).
- [PASS] `src/model_advanced.py` parses via `ast.parse` (no SyntaxError).

## Task 3. Manuscript word count

- [PASS] `wc -w manuscripts/manuscript.md` = 4,237 words. Target is 4,000-5,000. Within band.

## Task 4. Self-contained HTML

- [WARN] `grep -E 'href="http|src="http' deliverables/presentation.html` returns 1 hit:
  - `<a href="https://guillaumejaume.github.io/FUNSD/">guillaumejaume.github.io/FUNSD</a>` in the dataset row of the metadata table.
  - Not an external CSS / JS / image dependency, so the slide deck still renders fully offline; nevertheless it violates the "0 external resources" inline-only rule. Recommend converting the link to plain text or to `<a href="#">` with the URL preserved as visible label.

## Task 5. IMRaD completeness

Top-level headings present in `manuscripts/manuscript.md`:

- Title (H1)
- Abstract
- 1. Introduction
- 2. Methods
- 3. Results
- 4. Discussion
- 5. Conclusion
- References

- [PASS] All eight IMRaD-equivalent sections are present. Keywords block also present after Abstract.

## Task 6. Method drift

Methods named in manuscript section 2 vs presence in `src/model_baseline.py` or `src/model_advanced.py`:

- [PASS] BERT-base (bert-base-uncased): present in `model_baseline.py` (`MODEL_NAME = "bert-base-uncased"`).
- [PASS] LayoutLMv3-base (microsoft/layoutlmv3-base): present in `model_advanced.py` (`MODEL_NAME = "microsoft/layoutlmv3-base"`, `LayoutLMv3ForTokenClassification`, `LayoutLMv3Processor`).
- [PASS] BIO 7-tag scheme (O, B/I-question, B/I-answer, B/I-header): both scripts encode this label set.
- [PASS] WordPiece tokenizer max-length 512: handled by HuggingFace tokenizer in baseline.
- [PASS] AdamW + linear warmup 10%: both scripts set `WARMUP_RATIO = 0.1` and use HF Trainer (which uses AdamW with linear schedule by default).
- [PASS] Bounding-box normalisation to 0-1000 grid: `normalize_box` in `model_advanced.py`.
- [PASS] Image patches (ViT-style 16x16 patch encoder): handled by `LayoutLMv3Processor` and the LayoutLMv3 model itself.
- [PASS] Linking head MLP over `(q_emb, a_emb, q-a, box deltas)`: `LinkingHead` class plus `entity_centroid` / `pool_entity_embeddings` / `train_and_eval_linking_head` in `model_advanced.py`.
- [PASS] Negative-pair sampling (5:1 ratio): documented in section 2.4 and the linking-head training loop is present in `model_advanced.py`.
- [PASS] FP16 enabled when CUDA available, single-GPU training: documented in script comments.
- [PASS] LiLT, DocFormer, TILT, BROS, XYLayoutLM, LayoutXLM, Donut, Pix2Struct, UDOP, Tesseract: documented in section 2.7 as "alternatives not run", consistent with scaffold-only scope. No drift.

No method-drift findings.

## Task 7. Citation drift

Inline citations found in `manuscripts/manuscript.md` (unique keys): Appalaraju 2021, Cui 2021, Devlin 2019, Dosovitskiy 2021, Gu 2022, He 2016, Hong 2022, Huang 2022, Hwang 2021, Jaume 2019, Kim 2022, Kingma 2015, Lee 2023, Lewis 2020, Lin 2017, Liu 2019, Loshchilov 2019, Mathew 2021, Powalski 2021, Raffel 2020, Sang 2003, Smith 2007, Tang 2023, Wang 2022, Wolf 2020, Xu 2020, Xu 2021, plus one occurrence of `Xu 2022`.

Map vs `reports/references.md`:

- [PASS] All 27 of the unique keys above resolve to a numbered entry in `references.md` (checking author surname + publication year against the bibliography).
- [WARN] Orphan citation `[Xu 2022]` on manuscript line 94 (`**LayoutXLM** [Xu 2022, Wang 2022]. Multilingual LayoutLMv2 covering 53 languages.`). The references file lists LayoutXLM as entry 14: `Xu Y, ... LayoutXLM. arXiv. 2021. DOI:10.48550/arXiv.2104.08836` (year 2021, not 2022). Either the inline citation should read `[Xu 2021]` (matching the actual publication year) or the references file should be re-verified. The "quick map" block at the bottom of the manuscript already concedes this by noting `[Wang 2022] - LiLT, ACL 2022; also LayoutXLM, arXiv 2021`, but the inline `Xu 2022` is still mismatched.

## Task 8. Live CrossRef re-verification (5 random references)

DOIs picked: ref-1 (FUNSD), ref-6 (LayoutLMv3), ref-7 (LiLT), ref-21 (BERT), ref-26 (ResNet). All hit `https://api.crossref.org/works/{doi}` live.

- [PASS] DOI `10.1109/ICDARW.2019.10029` -> HTTP 200, title "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents". Matches references.md entry 1.
- [PASS] DOI `10.1145/3503161.3548112` -> HTTP 200, title "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking". Matches entry 6.
- [PASS] DOI `10.18653/v1/2022.acl-long.534` -> HTTP 200, title "LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding". Matches entry 7.
- [WARN] DOI `10.18653/v1/N19-1423` -> HTTP 200 (DOI resolves), but the CrossRef record has an empty `title` field (`['']`) and the `container-title` is "Proceedings of the 2019 Conference of the North". DOI is real (it is the canonical ACL Anthology DOI for BERT), this is a CrossRef metadata-quality issue not a reference fabrication. Recommend a one-line note in references.md or a switch to the arXiv DOI `10.48550/arXiv.1810.04805` which has full title metadata.
- [PASS] DOI `10.1109/CVPR.2016.90` -> HTTP 200, title "Deep Residual Learning for Image Recognition". Matches entry 26.

## Task 9. Em-dash scan

- [PASS] `grep -c "—"` across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, `src/model_baseline.py`, `src/model_advanced.py`, `manuscripts/manuscript.md`, `deliverables/presentation.html` returns 0 hits in every file. Total = 0.

## Task 10. AI-tell scan

- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude' .` returns no hits anywhere in the project tree.

## Task 11. Checkpoint schema

Keys in `checkpoint.json`: `project_number`, `title`, `methodology`, `phase`, `status`, `needs_main_session_execution`, `blockers`.

- [PASS] All four required keys present (`project_number`, `title`, `methodology`, `status`). `phase`, `needs_main_session_execution`, `blockers` are extra fields, allowed.

## Saved-model artefacts (extra check, not in the 11 task list)

Project 18 is in the #9-#21 range (scaffold-only per cohort plan), so no saved model is expected. Confirming:

- `deliverables/` contains only `presentation.html` (no `.pkl`, `.pt`, `.png`). [WARN-not-FAIL] consistent with scaffold-only status; brief.md and checkpoint flag `needs_main_session_execution: true`.

---

## Findings summary

- FAIL: 0
- WARN: 3 (one external href in HTML; one orphan `[Xu 2022]` citation; one CrossRef record with empty title for the BERT DOI)
- PASS: all other checks

Final status: PASS-WITH-WARNINGS. The project is structurally sound; the three warnings are small text fixes that do not block the main session executing the training scripts.
