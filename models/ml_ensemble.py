"""
Traditional ML Ensemble — Logistic Regression + Random Forest + XGBoost
Trained on TF-IDF vectorized features.
Outputs soft probability votes.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from typing import List, Tuple
import joblib


class MLEnsemble:
    def __init__(self, num_classes: int = 7):
        self.num_classes = num_classes
        self.le = LabelEncoder()

        lr  = LogisticRegression(max_iter=2000, C=5.0, solver="lbfgs")
        rf  = RandomForestClassifier(n_estimators=300, max_depth=None,
                                      min_samples_leaf=1, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="mlogloss", random_state=42, n_jobs=-1)

        self.voting = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
            voting="soft",
            weights=[2, 1, 2],   # LR and XGBoost weighted higher
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),          # up to trigrams
                max_features=20000,          # more features
                sublinear_tf=True,
                analyzer="word",
                min_df=1,
                strip_accents="unicode",
                token_pattern=r"(?u)\b\w+\b",
            )),
            ("clf", self.voting),
        ])

    def fit(self, texts: List[str], labels: List[str]):
        y = self.le.fit_transform(labels)
        self.pipeline.fit(texts, y)
        print(f"[ML Ensemble] Trained on {len(texts)} samples")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns (n_samples, num_classes) probability matrix."""
        probs = self.pipeline.predict_proba(texts)
        # Reorder columns to match global label order if needed
        return probs

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def save(self, path: str = "ml_ensemble.joblib"):
        joblib.dump({"pipeline": self.pipeline, "le": self.le}, path)
        print(f"[ML Ensemble] Saved to {path}")

    def load(self, path: str = "ml_ensemble.joblib"):
        obj = joblib.load(path)
        self.pipeline = obj["pipeline"]
        self.le       = obj["le"]
