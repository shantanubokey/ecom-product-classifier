"""
Data Augmentation for Hinglish product descriptions.
Techniques: synonym swap, random deletion, random swap, back-translation simulation.
More data = better accuracy, especially with only 15 samples/class.
"""

import random
from typing import List, Tuple

# Category-specific synonym pools
SYNONYMS = {
    "saree":       ["sari", "saari", "silk saree", "cotton saree"],
    "kurta":       ["kurti", "kurtha", "top"],
    "lehenga":     ["lehnga", "skirt set", "ghagra"],
    "necklace":    ["haar", "mala", "chain", "pendant"],
    "earrings":    ["jhumka", "jhumke", "ear studs", "drops"],
    "bangles":     ["chudi", "choodi", "kangan", "bracelet set"],
    "lipstick":    ["lip color", "lip stick", "lip shade"],
    "kajal":       ["kohl", "eye liner", "eyeliner"],
    "mehendi":     ["mehndi", "mehandi", "henna"],
    "chappal":     ["slipper", "sandal", "footwear"],
    "mobile":      ["phone", "smartphone", "handset"],
    "laptop":      ["notebook computer", "ultrabook", "computer"],
    "printer":     ["inkjet", "laser printer", "printing machine"],
    "rice":        ["chawal", "basmati", "grain"],
    "fertilizer":  ["khad", "manure", "urea"],
    "sofa":        ["couch", "settee", "seating"],
    "watch":       ["timepiece", "wristwatch", "ghadi"],
    "sunglasses":  ["shades", "goggles", "eyewear"],
    "notebook":    ["copy", "register", "diary"],
    "mixer":       ["grinder", "blender", "juicer mixer"],
}


def synonym_replace(text: str, n: int = 1) -> str:
    words = text.split()
    for _ in range(n):
        for i, w in enumerate(words):
            if w in SYNONYMS and random.random() < 0.5:
                words[i] = random.choice(SYNONYMS[w])
                break
    return " ".join(words)


def random_deletion(text: str, p: float = 0.15) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    return " ".join([w for w in words if random.random() > p]) or words[0]


def random_swap(text: str, n: int = 1) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def add_noise(text: str) -> str:
    """Simulate noisy Hinglish: random char repeat, extra spaces."""
    words = text.split()
    if words and random.random() < 0.3:
        idx = random.randint(0, len(words) - 1)
        w = words[idx]
        if len(w) > 2:
            pos = random.randint(1, len(w) - 1)
            words[idx] = w[:pos] + w[pos] + w[pos:]  # repeat one char
    return " ".join(words)


def augment_dataset(
    samples: List[Tuple[str, str]],
    augment_factor: int = 4,
) -> List[Tuple[str, str]]:
    """
    Augment each sample `augment_factor` times using random techniques.
    Returns original + augmented samples.
    """
    augmented = list(samples)
    techniques = [synonym_replace, random_deletion, random_swap, add_noise]

    for text, label in samples:
        for _ in range(augment_factor):
            fn = random.choice(techniques)
            new_text = fn(text)
            if new_text.strip() and new_text != text:
                augmented.append((new_text, label))

    random.shuffle(augmented)
    print(f"[Augment] {len(samples)} → {len(augmented)} samples ({augment_factor}x)")
    return augmented
