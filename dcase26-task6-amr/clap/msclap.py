"""Shared Microsoft MS-CLAP embedding helpers (audio arrays + text).

Mirrors ``clap.clap``'s API (``embed_audio_arrays`` / ``embed_texts``) so
``retrieval/clap_windows.py`` and ``retrieval/moment_retrieval.py`` can swap
backends via a flag. Model: ``microsoft/msclap`` 2023 checkpoint, 1024-d
(vs. LAION CLAP's 512-d -- callers must not assume a fixed dim across
backends), L2-normalized to match ``clap.clap``'s contract.

Unlike LAION CLAP (HF ``transformers``, array-in), the ``msclap`` package's
``CLAP.get_audio_embeddings()`` only accepts file paths -- it loads audio
itself via torchaudio. So the array-in embedder here writes each window to a
temp wav first. Batched across the whole list per call (not one file at a
time), so the temp-file overhead is one write per window, not per network
round trip.

Run in the ``audio-emb`` conda env (has ``msclap`` + a ``transformers``
release recent enough for the tokenizer patch below; the base env used
elsewhere in this repo does not have ``msclap`` installed).
"""

from __future__ import annotations

import os
import tempfile
import types

import numpy as np

MSCLAP_MODEL_VERSION = "2023"
EMB_DIM = 1024

_WRAPPER = None


def load_msclap():
    """Load + cache the MS-CLAP wrapper. Returns the ``msclap.CLAP`` instance."""
    global _WRAPPER
    if _WRAPPER is not None:
        return _WRAPPER
    import torch
    from msclap import CLAP

    wrapper = CLAP(version=MSCLAP_MODEL_VERSION, use_cuda=torch.cuda.is_available())

    # msclap 1.3.4 calls the tokenizer's deprecated `encode_plus` alias,
    # which recent `transformers` releases removed in favor of `__call__`.
    # Same patch as fiftyone-audio-toolkit's _load_msclap().
    if not hasattr(wrapper.tokenizer, "encode_plus"):

        def _encode_plus(self, **kwargs):
            return self(**kwargs)

        wrapper.tokenizer.encode_plus = types.MethodType(_encode_plus, wrapper.tokenizer)

    _WRAPPER = wrapper
    return _WRAPPER


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def embed_audio_arrays(arrays: list[np.ndarray], sr: int = 48000, **_ignored) -> np.ndarray:
    """Embed a list of mono float32 waveforms (any common sample rate; msclap
    resamples internally). ``sr`` is the rate the arrays are already at.

    Returns an ``(n, 1024)`` L2-normalized float32 array, row-aligned to input.
    """
    import soundfile as sf

    wrapper = load_msclap()
    tmp_paths = []
    try:
        for arr in arrays:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, np.asarray(arr, dtype=np.float32), sr)
            tmp_paths.append(path)
        if not tmp_paths:
            return np.empty((0, EMB_DIM), dtype=np.float32)
        emb = wrapper.get_audio_embeddings(tmp_paths, resample=True)
        emb = np.asarray(emb.detach().cpu().numpy(), dtype=np.float32)
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
    return _normalize(emb)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed text queries/captions. Returns ``(n, 1024)`` L2-normalized float32."""
    wrapper = load_msclap()
    if not texts:
        return np.empty((0, EMB_DIM), dtype=np.float32)
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i : i + batch_size])
        emb = wrapper.get_text_embeddings(chunk)
        out.append(emb.detach().cpu().numpy().astype(np.float32))
    return _normalize(np.concatenate(out, axis=0))
