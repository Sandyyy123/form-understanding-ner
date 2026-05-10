![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![NLP](https://img.shields.io/badge/Document-AI-purple) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# FUNSD Form Understanding — Document NER and Key-Value Linking

Token-level entity tagging (question/answer/header/other) and key-value linking on scanned business forms using LayoutLM.

---

## Task

**Document Intelligence (NER + Relation Extraction)**

---

## Architecture

```
Scanned Form → OCR + Bounding Box → LayoutLM Token Classification → Key-Value Pair Linking
```

---

## Key Features

- Token classification: question / answer / header / other (BIO tagging)
- LayoutLM (Microsoft) — text + position + image token embeddings
- BERT baseline for text-only comparison
- Key-value linking: which answer tokens belong to which question
- F1 score (entity-level) evaluation on FUNSD benchmark

---

## Dataset

[FUNSD: Form Understanding in Noisy Scanned Documents (ICDAR)](https://guillaumejaume.github.io/FUNSD/)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/form-understanding-ner.git
cd form-understanding-ner
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`transformers (LayoutLM) · pytesseract · PyTorch · seqeval`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Germany.

---

## License

MIT
