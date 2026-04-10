"""
LSTM Classifier — Sequential dependency model with Attention
Trained on BPE token IDs from the TikTok-style tokenizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
        context = (lstm_out * weights).sum(dim=1)         # (batch, hidden*2)
        return context


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 19,
                 num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention  = Attention(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)          # (batch, seq, hidden*2)
        context = self.attention(out)    # (batch, hidden*2)
        context = self.dropout(context)
        return self.classifier(context)


def train_lstm(model, dataloader, optimizer, criterion, device, epochs=10,
               scheduler=None):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for token_ids, labels in dataloader:
            token_ids = token_ids.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            logits = model(token_ids)
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[LSTM] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")


def predict_lstm(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for token_ids, _ in dataloader:
            token_ids = token_ids.to(device)
            logits    = model(token_ids)
            probs     = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    return all_probs, all_probs.argmax(axis=1)
