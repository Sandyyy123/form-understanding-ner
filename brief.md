# Project 18 - FUNSD Form Understanding

## One-line summary
Token-level entity tagging (`question` / `answer` / `header` / `other`) and key-value linking on scanned business forms, benchmarking a BERT-base text-only baseline against a LayoutLMv3 layout-aware model on the FUNSD dataset (199 forms, ~16 MB, Jaume 2019).

## Author
Sandeep Grover, Liora MLE Programme, Cohort 6973.

## Business context
Form understanding is the second-largest document-intelligence workload in DACH Mittelstand after invoice processing. Insurance, HR, legal, and clinical-trial workflows depend on extracting question-answer pairs from non-uniform forms (Schadenmeldung, Lohnsteuerbescheinigung, Krankenkasse-Anträge, CRF, Patient-intake). Manual transcription costs EUR 2-6 per page; intelligent document processing (IDP) targets >90% straight-through processing on header / question / answer entities, with the remaining minority routed to a human-in-the-loop reviewer. The same techniques apply to Behörden-Formulare under DSGVO and to multilingual variants via LayoutXLM and LiLT.

## Twin project
Project #18 (forms) is the structural twin of project #12 (SROIE receipts). Both use LayoutLMv3, both rely on bbox-aware token classification, and both publish per-entity F1 as the headline metric. The two together form the IDP backbone of the cohort portfolio.

## Task definition
- Input: a single form image (PNG) plus token-level annotations - text, bounding boxes, and word-level tokens (`words` field).
- Output: a label per token from `{B-question, I-question, B-answer, I-answer, B-header, I-header, O}` plus key-value linking edges between question and answer ids.
- Metric: per-entity F1 on the FUNSD test split (50 forms, ~14k tokens), reported per class and macro-averaged. Linking F1 reported separately on entity pairs.

## Methodology

### Baseline: `src/model_baseline.py`
- HuggingFace `bert-base-uncased` fine-tuned for token classification (text-only, layout ignored).
- 4-class entity tagging in BIO format (7 effective tags including the `O` tag).
- Tokens flattened from FUNSD `words` field into a single sequence per form, padded / truncated to 512.
- AdamW optimiser, 10 epochs, batch size 8, learning rate 5e-5, linear warmup 10%.
- Saves model and metrics to `deliverables/`.

### Advanced: `src/model_advanced.py`
- HuggingFace `microsoft/layoutlmv3-base` with bounding-box embeddings + image patches.
- Multi-task head: token classification (entity tag) + key-value linking edges scored as a binary classifier over question-answer id pairs.
- LayoutLMv3 normalises bboxes to a 0-1000 grid; image patches are 16x16 tokens of the page raster.
- LiLT (`SCUT-DLVCLab/lilt-roberta-en-base`) is documented in the script as a layout-only alternative that can be paired with any text encoder for multilingual transfer.
- DocFormer (Appalaraju 2021) and TILT (Powalski 2021) are mentioned as architectural cousins.
- AdamW, 5 epochs (LayoutLMv3 converges faster), batch size 4 (image patches dominate memory), learning rate 3e-5.

## Dataset
FUNSD direct download: <https://guillaumejaume.github.io/FUNSD/dataset.zip> (~16 MB). Extracted into `data/dataset/{training_data, testing_data}/{annotations, images}/`. 149 train + 50 test forms with `box`, `text`, `label`, `words`, `linking`, `id` per entity.

## Phase 1 status
Code-only scaffold. Data is downloaded. Notebooks and scripts are written but not run. Main session will execute `notebooks/01_EDA.ipynb`, `src/model_baseline.py`, `src/model_advanced.py` and patch result placeholders in the manuscript and presentation.
