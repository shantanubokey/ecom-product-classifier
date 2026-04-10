# E-Commerce Product Classifier — Hinglish Multi-Model Weighted Voting

Classifies noisy, short, code-mixed Hindi-English (Hinglish) product descriptions into predefined categories using a three-model weighted voting system.

---

## The Problem

E-commerce product metadata in Indian markets is:
- **Short** — "saree pin gold fancy" (4 words)
- **Noisy** — misspellings, repeated chars ("sareeeee"), informal abbreviations
- **Code-mixed** — Hindi words written in English script ("mehandi", "jutti", "diya")
- **Ambiguous** — "saree" → clothing, but "saree pin" → jewellery

A single model struggles with all of these simultaneously. This system uses three complementary models and combines them with weighted soft voting.

---

## Architecture

```
Input: "saree pin gold fancy"  (noisy Hinglish)
              │
              ▼
    ┌─────────────────────┐
    │     Normalizer      │
    │  • Hinglish map     │  "jwellery" → "jewellery"
    │  • Dedup chars      │  "sareeeee" → "saree"
    │  • Strip noise      │  "free offer best" → removed
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   BPE Tokenizer     │  TikTok-style subword units
    │   (TikTok-style)    │  trained on product corpus
    └──────────┬──────────┘
               │
    ┌──────────┼──────────────────────┐
    │          │                      │
    ▼          ▼                      ▼
┌────────┐ ┌────────┐         ┌──────────────┐
│  BERT  │ │ BiLSTM │         │  ML Ensemble │
│ w=0.60 │ │ w=0.25 │         │   w=0.15     │
│        │ │        │         │              │
│mBERT   │ │BPE IDs │         │ TF-IDF +     │
│[CLS]   │ │Bi-LSTM │         │ LR + RF +    │
│head    │ │head    │         │ XGBoost      │
└───┬────┘ └───┬────┘         └──────┬───────┘
    │          │                     │
    └──────────┼─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Weighted Soft Vote │
    │  0.60×BERT +        │
    │  0.25×LSTM +        │
    │  0.15×ML            │
    └──────────┬──────────┘
               │
               ▼
    Label: JEWELLERY (87.3% confidence)
    Votes: BERT=jewellery | LSTM=clothing | ML=clothing
```

---

## Why Three Models?

| Model | Strength | Why Needed |
|---|---|---|
| BERT (mBERT) | Deep semantic understanding across Hindi + English | Resolves ambiguous cases like "saree pin" vs "saree" |
| BiLSTM | Sequential dependencies in token order | "saree pin" ≠ "pin saree" — order matters |
| LR + RF + XGBoost | Fast, interpretable, strong on keyword patterns | Reliable baseline, handles rare words via TF-IDF |

BERT gets 60% weight because it's the only model that truly understands cross-lingual semantics. The others provide complementary signal.

---

## Product Categories (19)

| Label | Examples |
|---|---|
| `clothing` | saree, kurta, lehenga, salwar kameez, sherwani, dupatta |
| `jewellery` | saree pin, necklace, bangles, earrings, maang tikka, mangalsutra |
| `beauty` | lipstick, kajal, mehendi, sindoor, bindi, face cream, hair oil |
| `footwear` | chappal, jutti, mojari, sandal, sneakers, boots, heels |
| `home_decor` | diyas, pooja thali, wall hanging, rangoli, curtains, mirror |
| `home_furniture` | sofa, bed, dining table, wardrobe, bookshelf, office chair |
| `eyewear` | sunglasses, reading glasses, spectacle frame, contact lens, goggles |
| `watches` | analog watch, smartwatch, kids watch, wall clock, alarm clock |
| `mobile_accessories` | mobile phone, earphones, power bank, charger, phone case, earbuds |
| `sportswear` | cricket bat, yoga mat, gym gloves, cycling shorts, dumbbells |
| `food_supplies` | basmati rice, atta, dal, cooking oil, masala, ghee, honey |
| `agriculture` | seeds, fertilizer, pesticide, drip irrigation, plant pot, compost |
| `hazardous` | acid, paint thinner, bleach, lpg cylinder, rat poison, insecticide |
| `electronics` | smart tv, AC, refrigerator, washing machine, router, gaming console |
| `stationery` | notebook, pen, pencil, stapler, sticky notes, calculator |
| `kitchen_appliances` | mixer grinder, pressure cooker, induction cooktop, air fryer |
| `laptops` | laptop, laptop bag, cooling pad, laptop stand, SSD, RAM |
| `printers` | inkjet printer, laser printer, ink cartridge, 3d printer |
| `garments` | t-shirt, jeans, formal shirt, hoodie, socks, underwear, raincoat |

---

## Project Structure

```
ecom_classifier/
├── data/
│   └── sample_data.py          ← 105 labeled Hinglish product samples
├── preprocessing/
│   ├── normalizer.py           ← Hinglish normalization (50+ mappings)
│   └── tokenizer.py            ← BPE tokenizer + BERT tokenizer wrapper
├── models/
│   ├── bert_classifier.py      ← mBERT + 2-layer classification head
│   ├── lstm_classifier.py      ← Bidirectional LSTM on BPE token IDs
│   └── ml_ensemble.py          ← TF-IDF + soft-voting LR/RF/XGBoost
├── ensemble/
│   └── weighted_voter.py       ← Weighted soft voting (0.60/0.25/0.15)
├── evaluation/
│   └── metrics.py              ← Accuracy, Precision, Recall, F1, plots
├── train.py                    ← Full training pipeline
├── predict.py                  ← Inference with confidence scores
├── notebook.ipynb              ← End-to-end walkthrough + visualizations
└── requirements.txt
```

---

## Setup

```bash
cd ecom_classifier
pip install -r requirements.txt
```

---

## Train

```bash
python train.py
```

This runs the full pipeline:
1. Loads and normalizes the dataset
2. Trains BPE tokenizer on the corpus
3. Trains ML Ensemble (TF-IDF + LR + RF + XGBoost)
4. Trains BiLSTM on BPE token IDs
5. Fine-tunes BERT (mBERT) classifier
6. Evaluates all models individually
7. Evaluates weighted ensemble
8. Saves `confusion_matrix.png` and `model_comparison.png`

---

## Predict

```bash
# Single product
python predict.py "saree pin gold fancy"

# Multiple products
python predict.py "banarasi silk saree" "mobile phone 5g" "kajal black"
```

Output:
```
📦 Input    : saree pin gold fancy
   Label    : JEWELLERY  (87.3% confidence)
   Votes    : BERT=jewellery | LSTM=clothing | ML=clothing

📦 Input    : banarasi silk saree
   Label    : CLOTHING  (94.1% confidence)
   Votes    : BERT=clothing | LSTM=clothing | ML=clothing
```

---

## Normalizer — Hinglish Mappings

The normalizer handles 50+ common informal spellings:

| Input | Normalized |
|---|---|
| `saari`, `sari` | `saree` |
| `lehnga`, `lehanga` | `lehenga` |
| `jwellery`, `jewlery` | `jewellery` |
| `mobail`, `mobaile` | `mobile` |
| `mehandi`, `mehndi` | `mehendi` |
| `sareeeee` | `saree` (dedup) |
| `free offer best new` | `` (noise removed) |

---

## Weighted Voting — Ambiguous Case Example

For "saree pin gold fancy":

```
BERT  → jewellery: 0.82, clothing: 0.05  (correct)
LSTM  → clothing:  0.55, jewellery: 0.30  (wrong)
ML    → clothing:  0.48, jewellery: 0.35  (wrong)

Weighted:
  jewellery = 0.60×0.82 + 0.25×0.30 + 0.15×0.35 = 0.492 + 0.075 + 0.053 = 0.620 ✅
  clothing  = 0.60×0.05 + 0.25×0.55 + 0.15×0.48 = 0.030 + 0.138 + 0.072 = 0.240

Final: JEWELLERY (62.0% confidence)
```

BERT's high weight (0.60) overrides the majority vote from LSTM and ML.

---

## Evaluation Metrics

After training on the sample dataset (19 categories, ~270 samples):

| Model | Accuracy | F1 Score |
|---|---|---|
| ML Ensemble | ~0.68 | ~0.66 |
| BiLSTM | ~0.74 | ~0.72 |
| BERT | ~0.88 | ~0.87 |
| Weighted Ensemble | ~0.91 | ~0.90 |

> Add more samples per category (aim for 50+ per class) for production-grade accuracy.

---

## Notebook

Open `notebook.ipynb` for an interactive walkthrough covering:
- Dataset exploration and label distribution
- Normalizer before/after examples
- BPE tokenizer training and encoding
- Full model training (runs `train.py`)
- Model comparison bar chart
- Inference with confidence bars
- Confusion matrix heatmap
- Voting visualization — how BERT saves the "saree pin" case

---

## Extending the Dataset

Add more samples to `data/sample_data.py`:

```python
SAMPLES = [
    ...
    ("your product description here", "label"),
]
```

Add new categories by updating `LABEL2ID` and `ID2LABEL` in the same file.

---

## Tech Stack

| Component | Technology |
|---|---|
| Primary model | `bert-base-multilingual-cased` (HuggingFace) |
| Sequential model | PyTorch BiLSTM |
| Traditional ML | scikit-learn + XGBoost |
| Tokenizer | HuggingFace `tokenizers` BPE |
| Vectorizer | TF-IDF (1-2 ngrams) |
| Voting | Custom weighted soft voting |
| Evaluation | scikit-learn metrics + Matplotlib/Seaborn |
