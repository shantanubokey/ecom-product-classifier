"""
Hinglish Normalizer
Handles: informal spellings, code-mixed text, common abbreviations,
repeated characters, and noisy e-commerce product descriptions.
"""

import re
from typing import List

# Common Hinglish informal → standard mappings
HINGLISH_MAP = {
    "saari":   "saree",
    "sari":    "saree",
    "lehnga":  "lehenga",
    "lehanga": "lehenga",
    "kurtha":  "kurta",
    "kameej":  "kameez",
    "jwellery":"jewellery",
    "jwelry":  "jewellery",
    "jewlery":  "jewellery",
    "chappel": "chappal",
    "chappal": "chappal",
    "mobail":  "mobile",
    "mobaile": "mobile",
    "earfone": "earphone",
    "earfones":"earphones",
    "liptick": "lipstick",
    "kajjal":  "kajal",
    "mehandi": "mehendi",
    "mehndi":  "mehendi",
    "diya":    "diyas",
    "diwa":    "diwali",
    "puja":    "pooja",
    "pooja":   "pooja",
    "sindur":  "sindoor",
    "bindi":   "bindi",
    "tikka":   "tikka",
    "nath":    "nath",
    "payal":   "anklet",
    "bichiya": "toe ring",
    "kada":    "bracelet",
    "haar":    "necklace",
    "mala":    "necklace",
    "jhumka":  "earring",
    "jhumke":  "earring",
    "choodi":  "bangle",
    "chudi":   "bangle",
    "kangan":  "bangle",
    "jutti":   "jutti",
    "mojri":   "mojari",
    "khilona": "toy",
    "khilone": "toys",
    "gudiya":  "doll",
    "ghar":    "home",
    "kapda":   "clothing",
    "kapde":   "clothing",
}

# Units / noise patterns to clean
NOISE_PATTERNS = [
    r"\b\d+\s*(pcs|pc|pieces|piece|set|sets|pack|packs|nos|no)\b",
    r"\b(free|offer|sale|discount|buy|get|new|best|top|hot|trending)\b",
    r"[^\w\s]",   # remove punctuation
]


def normalize(text: str) -> str:
    text = text.lower().strip()

    # Remove repeated characters (e.g. "sareeeee" → "saree")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Apply Hinglish mappings
    tokens = text.split()
    tokens = [HINGLISH_MAP.get(t, t) for t in tokens]
    text = " ".join(tokens)

    # Remove noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_batch(texts: List[str]) -> List[str]:
    return [normalize(t) for t in texts]
