"""
BERT Classifier — Primary model (highest weight in ensemble)
Uses bert-base-multilingual-cased for Hinglish support.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List, Tuple
import numpy as np


class BERTClassifier(nn.Module):
    MODEL_NAME = "bert-base-multilingual-cased"

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(self.MODEL_NAME)
        hidden_size  = self.bert.config.hidden_size   # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        return logits


def train_bert(model, dataloader, optimizer, criterion, device, epochs=5,
               scheduler=None):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        if scheduler:
            scheduler.step()
        acc = correct / total
        print(f"[BERT] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")


def predict_bert(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities, predictions)."""
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    return all_probs, all_probs.argmax(axis=1)
