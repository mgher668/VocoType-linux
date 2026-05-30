"""Core runtime package for the VoCoType Linux IBus application."""

from vocotype_version import __version__
from .config import DEFAULT_CONFIG, ensure_logging_dir, load_config

__all__ = [
    "DEFAULT_CONFIG",
    "ensure_logging_dir",
    "load_config",
    "AudioCapture",
    "TranscriptionWorker",
    "TranscriptionResult",
    "__version__",
]


def __getattr__(name):
    if name == "AudioCapture":
        from .audio_capture import AudioCapture

        return AudioCapture
    if name in ("TranscriptionWorker", "TranscriptionResult"):
        from .transcribe import TranscriptionResult, TranscriptionWorker

        return {"TranscriptionWorker": TranscriptionWorker, "TranscriptionResult": TranscriptionResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
