"""
Weighted Soft Voting Ensemble
Combines BERT (primary), LSTM, and ML Ensemble probabilities.
BERT gets highest weight to resolve ambiguous Hinglish cases.
"""

import numpy as np
from typing import List, Dict, Tuple


# Default weights — BERT is primary
DEFAULT_WEIGHTS = {
    "bert": 0.60,
    "lstm": 0.25,
    "ml":   0.15,
}


class WeightedVoter:
    def __init__(self, weights: Dict[str, float] = None, id2label: Dict[int, str] = None):
        self.weights  = weights or DEFAULT_WEIGHTS
        self.id2label = id2label or {}
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def vote(
        self,
        bert_probs: np.ndarray,   # (n, num_classes)
        lstm_probs: np.ndarray,   # (n, num_classes)
        ml_probs:   np.ndarray,   # (n, num_classes)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            final_probs: (n, num_classes) weighted average probabilities
            predictions: (n,) predicted class indices
        """
        final_probs = (
            self.weights["bert"] * bert_probs +
            self.weights["lstm"] * lstm_probs +
            self.weights["ml"]   * ml_probs
        )
        predictions = final_probs.argmax(axis=1)
        return final_probs, predictions

    def vote_with_confidence(
        self,
        bert_probs: np.ndarray,
        lstm_probs: np.ndarray,
        ml_probs:   np.ndarray,
    ) -> List[Dict]:
        """
        Returns list of dicts with label, confidence, and per-model breakdown.
        """
        final_probs, predictions = self.vote(bert_probs, lstm_probs, ml_probs)
        results = []
        for i, pred in enumerate(predictions):
            label      = self.id2label.get(int(pred), str(pred))
            confidence = float(final_probs[i, pred])
            results.append({
                "label":      label,
                "confidence": round(confidence, 4),
                "all_scores": {
                    self.id2label.get(j, str(j)): round(float(final_probs[i, j]), 4)
                    for j in range(final_probs.shape[1])
                },
                "model_votes": {
                    "bert": self.id2label.get(int(bert_probs[i].argmax()), "?"),
                    "lstm": self.id2label.get(int(lstm_probs[i].argmax()), "?"),
                    "ml":   self.id2label.get(int(ml_probs[i].argmax()),   "?"),
                },
            })
        return results
