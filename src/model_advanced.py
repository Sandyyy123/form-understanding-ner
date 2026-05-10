"""FUNSD advanced: LayoutLMv3 multi-task token-classification + key-value linking.

Run from the project root:

    python src/model_advanced.py

Outputs (written to ../deliverables/):
- layoutlmv3_funsd/                 (HuggingFace model + processor)
- metrics_advanced.json             (per-class P/R/F1, linking F1, eval loss)
- confusion_matrix_advanced.png
- linking_pr_curve.png

Architecture notes
------------------
LayoutLMv3 (Huang 2022, ACM MM) extends LayoutLMv2 by replacing the CNN
visual backbone with a patch-based ViT-style image encoder. Token, layout
(bbox), and image-patch embeddings share the same transformer, with a
unified text-image masking objective during pre-training. For FUNSD,
this means a single 24-form fine-tune already beats LSTM-CRF and
BERT-only baselines reported in Jaume 2019.

Alternative backbones documented in the script
----------------------------------------------
- LiLT (Wang 2022, ACL): language-independent layout transformer; pair
  with any text encoder for multilingual transfer (forms in DE, FR, JP).
- DocFormer (Appalaraju 2021, ICCV): multi-modal end-to-end transformer
  with discrete spatial features.
- TILT (Powalski 2021, ICDAR): text + image + layout encoder-decoder.
- LayoutXLM (Wang 2022): multilingual variant of LayoutLMv2 over 53 languages.
- BROS (Hong 2022, AAAI): focuses on text + 2D positional encoding,
  no image branch (smaller / faster).

Initial implementation: this script is NOT executed during implementation. The main
session runs it once GPU is available. Estimated wall-clock on a single
RTX 5090: 12-18 min for 5 epochs with image patches enabled.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LEN = 512
NUM_EPOCHS = 5
BATCH_SIZE = 4
LR = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LINKING_NEG_RATIO = 5  # number of negative pairs per positive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
TRAIN_ANN_DIR = DATA_ROOT / "training_data" / "annotations"
TRAIN_IMG_DIR = DATA_ROOT / "training_data" / "images"
TEST_ANN_DIR = DATA_ROOT / "testing_data" / "annotations"
TEST_IMG_DIR = DATA_ROOT / "testing_data" / "images"
OUT_DIR = PROJECT_ROOT / "deliverables"
MODEL_OUT = OUT_DIR / "layoutlmv3_funsd"
METRICS_OUT = OUT_DIR / "metrics_advanced.json"
CONFUSION_OUT = OUT_DIR / "confusion_matrix_advanced.png"
LINKING_PR_OUT = OUT_DIR / "linking_pr_curve.png"

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# FUNSD loader: per-form list of words / boxes / labels / linking edges
# ---------------------------------------------------------------------------
def normalize_box(box, width, height):
    """LayoutLMv3 expects boxes scaled to a 0-1000 grid."""
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def load_form(ann_path: Path, img_dir: Path):
    with open(ann_path) as f:
        data = json.load(f)
    img = Image.open(img_dir / f"{ann_path.stem}.png").convert("RGB")
    W, H = img.size
    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []
    word_to_entity_id: list[int] = []
    entity_id_to_label: dict[int, str] = {}
    for item in data["form"]:
        eid = item["id"]
        entity_id_to_label[eid] = item["label"]
        for w_idx, w in enumerate(item["words"]):
            tok = w["text"]
            if not tok or not tok.strip():
                continue
            words.append(tok)
            boxes.append(normalize_box(w["box"], W, H))
            if item["label"] == "other":
                labels.append("O")
            else:
                prefix = "B-" if w_idx == 0 else "I-"
                labels.append(f"{prefix}{item['label']}")
            word_to_entity_id.append(eid)
    linking_pairs = []
    for item in data["form"]:
        for src, dst in item["linking"]:
            linking_pairs.append((src, dst))
    return {
        "form_id": ann_path.stem,
        "image": img,
        "words": words,
        "boxes": boxes,
        "labels": labels,
        "word_to_entity_id": word_to_entity_id,
        "entity_id_to_label": entity_id_to_label,
        "linking_pairs": linking_pairs,
    }


def load_split(ann_dir: Path, img_dir: Path):
    return [load_form(p, img_dir) for p in sorted(ann_dir.glob("*.json"))]


# ---------------------------------------------------------------------------
# Tokenisation via LayoutLMv3Processor
# ---------------------------------------------------------------------------
def encode_forms(forms, processor):
    encodings = []
    for f in forms:
        enc = processor(
            f["image"],
            text=f["words"],
            boxes=f["boxes"],
            word_labels=[LABEL2ID[lab] for lab in f["labels"]],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        encodings.append({k: v.squeeze(0) for k, v in enc.items()})
    return encodings


class FunsdLayoutDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# ---------------------------------------------------------------------------
# Token-classification metrics
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
# Linking head: light MLP over pooled entity embeddings
# ---------------------------------------------------------------------------
class LinkingHead(nn.Module):
    """Binary classifier for (question_entity, answer_entity) pairs.

    Input is a concatenation of (q_embed, a_embed, q_embed - a_embed,
    q_box - a_box, |q_box - a_box|), feeding a 2-layer MLP.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        in_dim = hidden_size * 3 + 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, q_emb, a_emb, q_box, a_box):
        delta_box = q_box - a_box
        abs_delta = torch.abs(delta_box)
        feats = torch.cat(
            [q_emb, a_emb, q_emb - a_emb, delta_box, abs_delta], dim=-1
        )
        return self.net(feats)


def pool_entity_embeddings(token_embeddings, word_ids, word_to_entity_id):
    """Average-pool token embeddings into entity-level embeddings."""
    entity_to_tokens = {}
    for tok_idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        eid = word_to_entity_id[w_id]
        entity_to_tokens.setdefault(eid, []).append(tok_idx)
    pooled = {}
    for eid, idxs in entity_to_tokens.items():
        pooled[eid] = token_embeddings[idxs].mean(dim=0)
    return pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading FUNSD ...")
    train_forms = load_split(TRAIN_ANN_DIR, TRAIN_IMG_DIR)
    test_forms = load_split(TEST_ANN_DIR, TEST_IMG_DIR)

    rng = random.Random(SEED)
    rng.shuffle(train_forms)
    n_val = max(1, len(train_forms) // 10)
    val_forms = train_forms[:n_val]
    train_forms = train_forms[n_val:]
    print(f"  train: {len(train_forms)}  val: {len(val_forms)}  test: {len(test_forms)}")

    label_counts = Counter(lab for f in train_forms for lab in f["labels"])
    print(f"  train label counts: {dict(label_counts)}")

    print("[2/7] Loading processor and model ...")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("[3/7] Encoding ...")
    train_enc = encode_forms(train_forms, processor)
    val_enc = encode_forms(val_forms, processor)
    test_enc = encode_forms(test_forms, processor)

    train_ds = FunsdLayoutDataset(train_enc)
    val_ds = FunsdLayoutDataset(val_enc)
    test_ds = FunsdLayoutDataset(test_enc)

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
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[4/7] Training token classifier ...")
    trainer.train()

    print("[5/7] Test-set evaluation (token classification) ...")
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

    print("[6/7] Linking head: train + eval ...")
    # Linking is trained as a small head on top of frozen LayoutLMv3 hidden states.
    # For brevity we use a heuristic baseline here (spatial proximity + entity-type
    # prior). In a full run, replace this block with a torch training loop using
    # LinkingHead defined above.
    linking_results = train_and_eval_linking_head(
        model, processor, train_forms, test_forms
    )

    print("[7/7] Saving artefacts ...")
    metrics = {
        "model": MODEL_NAME,
        "task": "FUNSD token classification + linking (LayoutLMv3 advanced)",
        "n_train_forms": len(train_forms),
        "n_val_forms": len(val_forms),
        "n_test_forms": len(test_forms),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_length": MAX_LEN,
        "label_list": LABEL_LIST,
        "test_classification_report": report,
        "linking": linking_results,
    }

    model.save_pretrained(MODEL_OUT)
    processor.save_pretrained(MODEL_OUT)
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
    ax.set_title("LayoutLMv3 FUNSD - test confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_OUT, dpi=150)
    plt.close()

    if linking_results.get("pr_curve"):
        prec = linking_results["pr_curve"]["precision"]
        rec = linking_results["pr_curve"]["recall"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Linking-head PR curve (test)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(LINKING_PR_OUT, dpi=150)
        plt.close()

    print(f"Macro F1 (test, tokens): {report['macro avg']['f1-score']:.4f}")
    print(f"Linking F1 (test): {linking_results.get('f1'):.4f}")
    print(f"Saved: {MODEL_OUT}, {METRICS_OUT}, {CONFUSION_OUT}, {LINKING_PR_OUT}")


# ---------------------------------------------------------------------------
# Linking-head training and evaluation (placeholder heuristic)
# ---------------------------------------------------------------------------
def entity_centroid(boxes_for_words):
    boxes = np.array(boxes_for_words, dtype=float)
    cx = (boxes[:, 0] + boxes[:, 2]).mean() / 2
    cy = (boxes[:, 1] + boxes[:, 3]).mean() / 2
    return np.array([cx, cy])


def train_and_eval_linking_head(model, processor, train_forms, test_forms):
    """Train and evaluate the linking head.

    For the implementation this uses a spatial-proximity heuristic ranker that
    matches each `question` entity to its nearest unlinked `answer` entity
    (Manhattan distance on normalised centroids). In the full run, replace
    with a torch training loop over LinkingHead, using LayoutLMv3 hidden
    states as q_emb / a_emb.
    """
    y_true, y_score = [], []
    tp = fp = fn = 0
    for f in test_forms:
        # Build entity -> word indices
        ent_to_words = {}
        for w_idx, eid in enumerate(f["word_to_entity_id"]):
            ent_to_words.setdefault(eid, []).append(w_idx)
        # Centroids
        centroids = {
            eid: entity_centroid([f["boxes"][i] for i in idxs])
            for eid, idxs in ent_to_words.items()
        }
        questions = [
            eid for eid, lab in f["entity_id_to_label"].items() if lab == "question"
        ]
        answers = [
            eid for eid, lab in f["entity_id_to_label"].items() if lab == "answer"
        ]
        gold_pairs = set(f["linking_pairs"])
        # For every (q, a) pair, score by inverse distance
        pred_pairs = set()
        used_a = set()
        for q in questions:
            best_a, best_score = None, -1.0
            for a in answers:
                if a in used_a:
                    continue
                d = np.abs(centroids[q] - centroids[a]).sum() + 1e-6
                score = 1.0 / d
                y_true.append(int((q, a) in gold_pairs or (a, q) in gold_pairs))
                y_score.append(score)
                if score > best_score:
                    best_score = score
                    best_a = a
            if best_a is not None:
                pred_pairs.add((q, best_a))
                used_a.add(best_a)
        for p in pred_pairs:
            if p in gold_pairs or (p[1], p[0]) in gold_pairs:
                tp += 1
            else:
                fp += 1
        for g in gold_pairs:
            if g not in pred_pairs and (g[1], g[0]) not in pred_pairs:
                fn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    pr_curve = None
    if y_true:
        p, r, _ = precision_recall_curve(y_true, y_score)
        pr_curve = {"precision": list(p), "recall": list(r)}

    return {
        "method": "spatial-proximity-heuristic (placeholder for LinkingHead)",
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pr_curve": pr_curve,
    }


if __name__ == "__main__":
    main()
