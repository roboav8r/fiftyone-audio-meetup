"""Audio → visual helpers for the Audio + AI meetup demo.

FiftyOne has no native *audio* media type (a ``.wav`` sample is media_type
``unknown``). We bridge that gap two ways, each chosen to light up as much
*native* FiftyOne machinery as possible:

* :func:`mel_spectrogram_png` — a static mel-spectrogram PNG. Used as the
  precomputed thumbnail for the custom spectrogram sample renderer (and as a
  plain-image fallback on deployments without the renderer).

* :func:`spectrogram_video` — a spectrogram *video* with the original audio
  muxed back in (via ffmpeg ``showspectrumpic`` looped under the audio track).
  This turns a long audio clip into a ``media_type="video"`` sample, which
  gives us — for free — the native video timeline, audio playback,
  ``TemporalDetection`` moment overlays, and ActivityNet temporal evaluation.

Both are deterministic: same input + params → same output bytes (no clocks,
no RNG), so re-running a loader is idempotent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Static mel spectrogram (PNG)
# ---------------------------------------------------------------------------


def mel_spectrogram_png(
    audio_path: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    sr: int | None = None,
    n_mels: int = 128,
    fmax: int | None = None,
    width_in: float = 4.0,
    height_in: float = 2.0,
    dpi: int = 200,
    cmap: str = "magma",
) -> str:
    """Render ``audio_path`` to a borderless mel-spectrogram PNG at ``out_path``.

    The image is drawn edge-to-edge (no axes/margins) so its x-axis maps
    linearly to time — useful when it backs a video whose timeline scrubs
    left→right. Returns ``out_path``.
    """
    # Imported lazily so the module is importable without a display / before
    # the heavy audio stack is needed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    out_path = os.fspath(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(os.fspath(audio_path), sr=sr, mono=True)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    librosa.display.specshow(mels_db, sr=sr, ax=ax, cmap=cmap)
    fig.savefig(out_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Spectrogram video (MP4 with original audio muxed in)
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (conda-forge ffmpeg) before "
            "rendering spectrogram videos."
        )
    return path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed (exit %d):\n%s" % (proc.returncode, proc.stderr[-2000:])
        )


def _probe_duration(audio_path: str | os.PathLike) -> float:
    """Return the duration of ``audio_path`` in seconds via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe not found on PATH (ships with ffmpeg).")
    proc = subprocess.run(
        [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", os.fspath(audio_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe could not read duration of {audio_path}")
    return float(proc.stdout.strip())


def spectrogram_video(
    audio_path: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    size: str = "1280x480",
    color: str = "intensity",
    scale: str = "log",
    legend: bool = False,
    fps: int = 10,
    crf: int = 23,
) -> str:
    """Render a spectrogram video for ``audio_path`` with the audio muxed in.

    A single full-clip spectrogram image (ffmpeg ``showspectrumpic``) is held
    for the clip's duration while the original audio plays, so the resulting
    ``.mp4`` has the same length as the source audio and a time axis that lines
    up with FiftyOne's timeline. ``legend=False`` (the default) draws the
    spectrogram edge-to-edge so x maps cleanly to time.

    Returns ``out_path``.
    """
    ffmpeg = _ffmpeg()
    out_path = os.fspath(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    legend_opt = "enabled" if legend else "disabled"
    tmp_png = out_path + ".spec.png"
    duration = _probe_duration(audio_path)

    # 1) full-clip spectrogram → single image
    _run(
        [
            ffmpeg, "-y", "-i", os.fspath(audio_path),
            "-lavfi",
            f"showspectrumpic=s={size}:legend={legend_opt}:color={color}:scale={scale}",
            tmp_png,
        ]
    )
    # 2) loop the image under the original audio. We cap with an explicit
    #    "-t <duration>" because "-shortest" does not reliably terminate a
    #    "-loop 1" image input (the video stream over-runs the audio).
    try:
        _run(
            [
                ffmpeg, "-y",
                "-loop", "1", "-framerate", str(fps), "-i", tmp_png,
                "-i", os.fspath(audio_path),
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-r", str(fps), "-crf", str(crf),
                "-c:a", "aac", "-b:a", "128k",
                "-t", f"{duration:.3f}", "-shortest", out_path,
            ]
        )
    finally:
        if os.path.exists(tmp_png):
            os.remove(tmp_png)
    return out_path


def scrolling_spectrogram_video(
    audio_path: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    size: str = "1280x480",
    color: str = "intensity",
    scale: str = "log",
    fps: int = 25,
    crf: int = 23,
) -> str:
    """Alternative style: a live *scrolling* spectrum (ffmpeg ``showspectrum``).

    Eye-candy variant where the spectrum scrolls as the audio plays. The
    static :func:`spectrogram_video` is preferred for moment-retrieval because
    its fixed time axis aligns with temporal-detection overlays.
    """
    ffmpeg = _ffmpeg()
    out_path = os.fspath(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg, "-y", "-i", os.fspath(audio_path),
            "-filter_complex",
            f"[0:a]showspectrum=s={size}:mode=combined:slide=scroll:"
            f"color={color}:scale={scale}[v]",
            "-map", "[v]", "-map", "0:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), "-crf", str(crf),
            "-c:a", "aac", "-b:a", "128k", "-shortest", out_path,
        ]
    )
    return out_path
