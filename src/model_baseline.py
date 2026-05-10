"""FUNSD baseline: BERT-base token classification, layout ignored.

Run from the project root:

    python src/model_baseline.py

Outputs (written to ../deliverables/):
- bert_funsd_baseline/ (HuggingFace model + tokenizer)
- metrics_baseline.json (per-class P/R/F1, support, macro / micro F1, eval loss)
- confusion_matrix_baseline.png

Initial implementation: this script is NOT executed during implementation. The main session
runs it once GPU is available. Estimated wall-clock on a single RTX 5090: 6-9 min
for 10 epochs over 149 forms.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
NUM_EPOCHS = 10
BATCH_SIZE = 8
LR = 5e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
TRAIN_DIR = DATA_ROOT / "training_data" / "annotations"
TEST_DIR = DATA_ROOT / "testing_data" / "annotations"
OUT_DIR = PROJECT_ROOT / "deliverables"
MODEL_OUT = OUT_DIR / "bert_funsd_baseline"
METRICS_OUT = OUT_DIR / "metrics_baseline.json"
CONFUSION_OUT = OUT_DIR / "confusion_matrix_baseline.png"

# BIO label set: 4 entity classes -> 7 effective tags including O
LABEL_LIST = [
    "O",
    "B-question",
    "I-question",
    "B-answer",
    "I-answer",
    "B-header",
    "I-header",
]
LABEL2ID = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Load FUNSD into per-form (words, labels) sequences
# ---------------------------------------------------------------------------
def load_funsd_split(ann_dir: Path) -> list[dict]:
    """Return a list of {form_id, words, word_labels} dicts.

    word_labels are BIO tags. Entity label `other` is mapped to all-O (no entity).
    """
    forms = []
    for path in sorted(ann_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        words: list[str] = []
        labels: list[str] = []
        for item in data["form"]:
            entity_label = item["label"]
            for w_idx, w in enumerate(item["words"]):
                tok = w["text"]
                if not tok or not tok.strip():
                    continue
                words.append(tok)
                if entity_label == "other":
                    labels.append("O")
                else:
                    prefix = "B-" if w_idx == 0 else "I-"
                    labels.append(f"{prefix}{entity_label}")
        forms.append({"form_id": path.stem, "words": words, "word_labels": labels})
    return forms


# ---------------------------------------------------------------------------
# Tokenize and align labels to subword pieces
# ---------------------------------------------------------------------------
def encode(forms: list[dict], tokenizer):
    encoded = tokenizer(
        [f["words"] for f in forms],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_offsets_mapping=False,
    )
    aligned_labels = []
    for i, f in enumerate(forms):
        word_ids = encoded.word_ids(batch_index=i)
        prev_w = None
        seq = []
        for w_id in word_ids:
            if w_id is None:
                seq.append(-100)  # special tokens are ignored in loss
            elif w_id != prev_w:
                seq.append(LABEL2ID[f["word_labels"][w_id]])
            else:
                # subword continuation: keep entity but switch B->I
                lab = f["word_labels"][w_id]
                if lab.startswith("B-"):
                    lab = "I-" + lab[2:]
                seq.append(LABEL2ID[lab])
            prev_w = w_id
        aligned_labels.append(seq)
    encoded["labels"] = aligned_labels
    return encoded


class FunsdDataset(Dataset):
    def __init__(self, encoded):
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = encoded["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    flat_preds, flat_labels = [], []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            flat_preds.append(int(p))
            flat_labels.append(int(l))

    report = classification_report(
        flat_labels,
        flat_preds,
        labels=list(range(len(LABEL_LIST))),
        target_names=LABEL_LIST,
        zero_division=0,
        output_dict=True,
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading FUNSD splits ...")
    train_forms = load_funsd_split(TRAIN_DIR)
    test_forms = load_funsd_split(TEST_DIR)

    # 90/10 stratified-by-form val split
    rng = random.Random(SEED)
    rng.shuffle(train_forms)
    n_val = max(1, len(train_forms) // 10)
    val_forms = train_forms[:n_val]
    train_forms = train_forms[n_val:]
    print(f"  train: {len(train_forms)}  val: {len(val_forms)}  test: {len(test_forms)}")

    label_counts = Counter(lab for f in train_forms for lab in f["word_labels"])
    print(f"  train label counts: {dict(label_counts)}")

    print("[2/6] Tokenizer / model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("[3/6] Encoding ...")
    train_ds = FunsdDataset(encode(train_forms, tokenizer))
    val_ds = FunsdDataset(encode(val_forms, tokenizer))
    test_ds = FunsdDataset(encode(test_forms, tokenizer))

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("[4/6] Training ...")
    trainer.train()

    print("[5/6] Test-set evaluation ...")
    test_pred = trainer.predict(test_ds)
    preds = np.argmax(test_pred.predictions, axis=-1)
    labels = test_pred.label_ids

    flat_preds, flat_labels = [], []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            flat_preds.append(int(p))
            flat_labels.append(int(l))

    report = classification_report(
        flat_labels,
        flat_preds,
        labels=list(range(len(LABEL_LIST))),
        target_names=LABEL_LIST,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(LABEL_LIST))))

    metrics = {
        "model": MODEL_NAME,
        "task": "FUNSD token classification (text-only baseline)",
        "n_train_forms": len(train_forms),
        "n_val_forms": len(val_forms),
        "n_test_forms": len(test_forms),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_length": MAX_LEN,
        "label_list": LABEL_LIST,
        "test_classification_report": report,
    }

    print("[6/6] Saving model and metrics ...")
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_LIST,
        yticklabels=LABEL_LIST,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("BERT-base FUNSD baseline - test confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_OUT, dpi=150)
    plt.close()

    print(f"Macro F1 (test): {report['macro avg']['f1-score']:.4f}")
    print(f"Saved: {MODEL_OUT}, {METRICS_OUT}, {CONFUSION_OUT}")


if __name__ == "__main__":
    main()
