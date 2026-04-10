"""
Inference — classify new product descriptions
Usage: python predict.py "saree pin gold fancy"
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from data.sample_data         import LABEL2ID, ID2LABEL, NUM_CLASSES
from preprocessing.normalizer import normalize_batch
from preprocessing.tokenizer  import BPETokenizer, BERTTokenizerWrapper
from models.bert_classifier   import BERTClassifier
from models.lstm_classifier   import LSTMClassifier
from models.ml_ensemble       import MLEnsemble
from ensemble.weighted_voter  import WeightedVoter

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 32


def load_models():
    # BERT
    bert = BERTClassifier(num_classes=NUM_CLASSES)
    bert.load_state_dict(torch.load("bert_model.pt", map_location=DEVICE))
    bert.eval().to(DEVICE)

    # LSTM + BPE tokenizer
    bpe = BPETokenizer(save_path="bpe_tokenizer.json")
    bpe.load()
    lstm = LSTMClassifier(vocab_size=bpe.vocab_size_actual, num_classes=NUM_CLASSES)
    lstm.load_state_dict(torch.load("lstm_model.pt", map_location=DEVICE))
    lstm.eval().to(DEVICE)

    # ML Ensemble
    ml = MLEnsemble()
    ml.load("ml_ensemble.joblib")

    return bert, lstm, bpe, ml


def predict(texts: list, show_confidence: bool = True) -> list:
    texts_clean = normalize_batch(texts)

    bert, lstm, bpe, ml = load_models()
    voter = WeightedVoter(id2label=ID2LABEL)

    # BERT probs
    bert_tok  = BERTTokenizerWrapper()
    enc       = bert_tok.encode(texts_clean, max_length=MAX_LEN)
    with torch.no_grad():
        logits = bert(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        bert_probs = torch.softmax(logits, dim=1).cpu().numpy()

    # LSTM probs
    ids = torch.tensor(bpe.encode_batch(texts_clean, MAX_LEN), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = lstm(ids)
        lstm_probs = torch.softmax(logits, dim=1).cpu().numpy()

    # ML probs (aligned to global label order)
    raw_ml = ml.predict_proba(texts_clean)
    ml_probs = np.zeros((len(texts_clean), NUM_CLASSES))
    for i, cls_name in enumerate(ml.le.classes_):
        gidx = LABEL2ID.get(cls_name)
        if gidx is not None:
            ml_probs[:, gidx] = raw_ml[:, i]

    if show_confidence:
        return voter.vote_with_confidence(bert_probs, lstm_probs, ml_probs)
    else:
        _, preds = voter.vote(bert_probs, lstm_probs, ml_probs)
        return [ID2LABEL[p] for p in preds]


if __name__ == "__main__":
    test_inputs = sys.argv[1:] if len(sys.argv) > 1 else [
        "saree pin gold fancy",
        "banarasi silk saree red",
        "mobile phone 5g android",
        "kajal black waterproof",
        "diya clay diwali",
        "jutti punjabi embroidered",
        "soft toy teddy bear",
    ]

    print("\n🔍 Product Classification Results\n" + "="*50)
    results = predict(test_inputs)
    for text, res in zip(test_inputs, results):
        print(f"\n📦 Input    : {text}")
        print(f"   Label    : {res['label'].upper()}  ({res['confidence']*100:.1f}% confidence)")
        print(f"   Votes    : BERT={res['model_votes']['bert']} | LSTM={res['model_votes']['lstm']} | ML={res['model_votes']['ml']}")
