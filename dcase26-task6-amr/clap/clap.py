"""Shared LAION-CLAP embedding helpers (audio arrays + text).

Same model as John's audio toolkit / audio-ai-meetup so embeddings live in one
space across the panel, the threshold baselines, and the MCAP timeseries:
``laion/larger_clap_music_and_speech`` @ 48 kHz, 512-d, L2-normalized.

Unlike ``curation/embed_clap.py`` (whole-file, path-in), this exposes an
**array-in** embedder so callers can feed arbitrary sliding windows.
"""

from __future__ import annotations

import numpy as np

CLAP_MODEL_NAME = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48000
EMB_DIM = 512

_BUNDLE = None  # (model, processor, device)


def load_clap(device: str | None = None):
    """Load + cache the CLAP model/processor. Returns (model, processor, device)."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    import torch
    from transformers import ClapModel, ClapProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(device).eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
    _BUNDLE = (model, processor, device)
    return _BUNDLE


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _to_tensor(out, keys):
    """Unwrap a HF ModelOutput/tensor to the projected 2-D embedding tensor.

    transformers 5.x returns a ModelOutput from get_{audio,text}_features; older
    versions returned the tensor directly. Prefer the projected embed field.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for k in keys + ["pooler_output", "last_hidden_state"]:
        v = getattr(out, k, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"cannot extract embedding tensor from {type(out).__name__}")


def embed_audio_arrays(
    arrays: list[np.ndarray], batch_size: int = 32, device: str | None = None
) -> np.ndarray:
    """Embed a list of mono float32 waveforms (each already at 48 kHz).

    Returns an ``(n, 512)`` L2-normalized float32 array, row-aligned to input.
    Each array is treated as one CLAP input (the feature extractor pads/truncates
    to the model's ~10 s window), so pass ~10 s windows for moment retrieval.
    """
    import torch

    model, processor, device = load_clap(device)
    out: list[np.ndarray] = []
    for i in range(0, len(arrays), batch_size):
        chunk = [np.asarray(a, dtype=np.float32) for a in arrays[i : i + batch_size]]
        inputs = processor(
            audio=chunk, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            feats = _to_tensor(model.get_audio_features(**inputs), ["audio_embeds"])
        out.append(feats.detach().cpu().numpy().astype(np.float32))
    if not out:
        return np.empty((0, EMB_DIM), dtype=np.float32)
    return _normalize(np.concatenate(out, axis=0))


def embed_texts(
    texts: list[str], batch_size: int = 64, device: str | None = None
) -> np.ndarray:
    """Embed text queries/captions. Returns ``(n, 512)`` L2-normalized float32."""
    import torch

    model, processor, device = load_clap(device)
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = processor(text=chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = _to_tensor(model.get_text_features(**inputs), ["text_embeds"])
        out.append(feats.detach().cpu().numpy().astype(np.float32))
    if not out:
        return np.empty((0, EMB_DIM), dtype=np.float32)
    return _normalize(np.concatenate(out, axis=0))
