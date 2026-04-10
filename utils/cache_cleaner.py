"""
Cache & Memory Cleaner
Frees GPU VRAM, Python object cache, module cache, and system RAM.
Call clean() between training steps to keep memory usage low.
"""

import gc
import sys
import os


def clean_torch():
    """Free GPU VRAM and PyTorch caches."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            freed = torch.cuda.memory_reserved() / 1024**2
            print(f"[Cache] GPU cache cleared ({freed:.0f} MB reserved)")
        else:
            print("[Cache] No GPU — skipping VRAM flush")
    except ImportError:
        pass


def clean_python():
    """Run Python garbage collector and clear reference cycles."""
    before = _get_ram_mb()
    collected = gc.collect(generation=2)
    after = _get_ram_mb()
    print(f"[Cache] GC collected {collected} objects | RAM: {before:.0f} → {after:.0f} MB")


def clean_modules(prefixes: list = None):
    """
    Remove cached module imports from sys.modules.
    Useful when you've edited a module and want the notebook to reload it.
    Default prefixes: all local project modules.
    """
    prefixes = prefixes or [
        "data.", "preprocessing.", "models.", "ensemble.",
        "evaluation.", "utils.", "train", "predict",
    ]
    removed = []
    for key in list(sys.modules.keys()):
        if any(key.startswith(p) for p in prefixes):
            del sys.modules[key]
            removed.append(key)
    if removed:
        print(f"[Cache] Removed {len(removed)} cached modules: {removed[:5]}{'...' if len(removed)>5 else ''}")
    else:
        print("[Cache] No project modules in cache")


def clean_matplotlib():
    """Close all open matplotlib figures to free memory."""
    try:
        import matplotlib.pyplot as plt
        n = len(plt.get_fignums())
        plt.close("all")
        print(f"[Cache] Closed {n} matplotlib figures")
    except ImportError:
        pass


def clean_transformers():
    """Clear HuggingFace tokenizer/model cache from memory."""
    try:
        import transformers
        transformers.utils.logging.set_verbosity_error()
    except ImportError:
        pass
    # Remove cached tokenizer objects from sys.modules
    for key in list(sys.modules.keys()):
        if "transformers" in key and "cache" in key:
            del sys.modules[key]


def _get_ram_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def report():
    """Print current memory usage."""
    ram = _get_ram_mb()
    print(f"\n{'='*45}")
    print(f"  RAM usage : {ram:.1f} MB")
    try:
        import torch
        if torch.cuda.is_available():
            alloc   = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU alloc : {alloc:.1f} MB")
            print(f"  GPU reserv: {reserved:.1f} MB")
    except ImportError:
        pass
    print(f"  sys.modules: {len(sys.modules)} loaded")
    print(f"{'='*45}\n")


def clean(modules: bool = True, torch: bool = True,
          matplotlib: bool = True, verbose: bool = True):
    """
    Run all cleaners in one call.

    Args:
        modules:    reload project module cache
        torch:      free GPU VRAM
        matplotlib: close open figures
        verbose:    print memory report before/after
    """
    if verbose:
        print("\n[Cache] Before cleanup:")
        report()

    if modules:
        clean_modules()
    if torch:
        clean_torch()
    if matplotlib:
        clean_matplotlib()

    clean_python()   # always run GC last

    if verbose:
        print("\n[Cache] After cleanup:")
        report()
