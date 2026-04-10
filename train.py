"""
Full Training Pipeline — with augmentation, LR scheduling, label smoothing
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

from data.sample_data        import get_dataframe, LABEL2ID, ID2LABEL, NUM_CLASSES, SAMPLES
from data.augment            import augment_dataset
from preprocessing.normalizer import normalize_batch
from preprocessing.tokenizer  import BPETokenizer, BERTTokenizerWrapper
from models.bert_classifier   import BERTClassifier, train_bert, predict_bert
from models.lstm_classifier   import LSTMClassifier, train_lstm, predict_lstm
from models.ml_ensemble       import MLEnsemble
from ensemble.weighted_voter  import WeightedVoter
from evaluation.metrics       import evaluate, plot_confusion_matrix, compare_models

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 48       # longer context
BATCH_SIZE  = 16
BERT_EPOCHS = 8        # more epochs
LSTM_EPOCHS = 25


class BERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    print(f"[Train] Device: {DEVICE}")

    # ── Load, augment & preprocess ─────────────────────────────────────────────
    augmented = augment_dataset(SAMPLES, augment_factor=5)
    texts_raw  = [s[0] for s in augmented]
    labels_str = [s[1] for s in augmented]

    texts  = normalize_batch(texts_raw)
    labels = [LABEL2ID[l] for l in labels_str]
    label_names = list(LABEL2ID.keys())

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"[Train] Train: {len(X_train)} | Test: {len(X_test)} | Classes: {NUM_CLASSES}")

    results = {}

    # ── 1. ML Ensemble ─────────────────────────────────────────────────────────
    print("\n[1/3] Training ML Ensemble...")
    ml = MLEnsemble(num_classes=NUM_CLASSES)
    train_labels_str = [ID2LABEL[y] for y in y_train]
    ml.fit(X_train, train_labels_str)
    ml.save("ml_ensemble.joblib")

    ml_probs_test = ml.predict_proba(X_test)
    ml_preds_mapped = [LABEL2ID.get(ml.le.inverse_transform([p])[0], p)
                       for p in ml_probs_test.argmax(axis=1)]
    results["ML Ensemble"] = evaluate(y_test, ml_preds_mapped, label_names)

    # ── 2. BPE + LSTM ──────────────────────────────────────────────────────────
    print("\n[2/3] Training LSTM with Attention...")
    bpe = BPETokenizer(vocab_size=6000, save_path="bpe_tokenizer.json")
    bpe.train(X_train)

    train_ids = torch.tensor(bpe.encode_batch(X_train, MAX_LEN), dtype=torch.long)
    test_ids  = torch.tensor(bpe.encode_batch(X_test,  MAX_LEN), dtype=torch.long)
    train_y   = torch.tensor(y_train, dtype=torch.long)
    test_y    = torch.tensor(y_test,  dtype=torch.long)

    lstm_train_dl = DataLoader(TensorDataset(train_ids, train_y),
                               batch_size=BATCH_SIZE, shuffle=True)
    lstm_test_dl  = DataLoader(TensorDataset(test_ids, test_y),
                               batch_size=BATCH_SIZE)

    lstm = LSTMClassifier(vocab_size=bpe.vocab_size_actual, num_classes=NUM_CLASSES)
    opt  = torch.optim.AdamW(lstm.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=LSTM_EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_lstm(lstm, lstm_train_dl, opt, crit, DEVICE, epochs=LSTM_EPOCHS,
               scheduler=sched)
    torch.save(lstm.state_dict(), "lstm_model.pt")

    lstm_probs_test, lstm_preds_test = predict_lstm(lstm, lstm_test_dl, DEVICE)
    results["LSTM"] = evaluate(y_test, lstm_preds_test.tolist(), label_names)

    # ── 3. BERT ────────────────────────────────────────────────────────────────
    print("\n[3/3] Fine-tuning BERT...")
    bert_tok  = BERTTokenizerWrapper()
    train_enc = bert_tok.encode(X_train, max_length=MAX_LEN)
    test_enc  = bert_tok.encode(X_test,  max_length=MAX_LEN)

    bert_train_dl = DataLoader(BERTDataset(train_enc, y_train),
                               batch_size=BATCH_SIZE, shuffle=True)
    bert_test_dl  = DataLoader(BERTDataset(test_enc,  y_test),
                               batch_size=BATCH_SIZE)

    bert = BERTClassifier(num_classes=NUM_CLASSES)
    opt  = AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
    sched_bert = CosineAnnealingLR(opt, T_max=BERT_EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    train_bert(bert, bert_train_dl, opt, crit, DEVICE, epochs=BERT_EPOCHS,
               scheduler=sched_bert)
    torch.save(bert.state_dict(), "bert_model.pt")

    bert_probs_test, bert_preds_test = predict_bert(bert, bert_test_dl, DEVICE)
    results["BERT"] = evaluate(y_test, bert_preds_test.tolist(), label_names)

    # ── 4. Weighted Voting ─────────────────────────────────────────────────────
    print("\n[Ensemble] Weighted Voting (BERT=0.6, LSTM=0.25, ML=0.15)...")
    ml_probs_aligned = np.zeros((len(X_test), NUM_CLASSES))
    for i, cls_name in enumerate(ml.le.classes_):
        gidx = LABEL2ID.get(cls_name)
        if gidx is not None:
            ml_probs_aligned[:, gidx] = ml_probs_test[:, i]

    voter = WeightedVoter(id2label=ID2LABEL)
    final_probs, final_preds = voter.vote(bert_probs_test, lstm_probs_test,
                                          ml_probs_aligned)
    results["Weighted Ensemble"] = evaluate(y_test, final_preds.tolist(), label_names)

    # ── 5. Plots ───────────────────────────────────────────────────────────────
    compare_models(results, save_path="model_comparison.png")
    plot_confusion_matrix(y_test, final_preds.tolist(), label_names,
                          title="Weighted Ensemble — Confusion Matrix",
                          save_path="confusion_matrix.png")

    print("\n✅ Training complete.")
    return results


if __name__ == "__main__":
    main()
