"""Dependency checks for the ModernBERT training scripts."""

from __future__ import annotations


def require_transformers_4() -> None:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Install with: python -m pip install -r requirements-modernbert.txt"
        ) from exc

    version = getattr(transformers, "__version__", "0")
    major = _major_version(version)
    if major >= 5:
        raise RuntimeError(
            "transformers 5.x currently segfaults with this fast-tokenizer pipeline in this environment. "
            "Downgrade with: python -m pip install --upgrade --force-reinstall -r requirements-modernbert.txt"
        )


def hard_exit_success() -> None:
    """Exit after flushing streams, bypassing fast-tokenizer shutdown crashes."""
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _major_version(version: str) -> int:
    head = version.split(".", 1)[0]
    digits = "".join(char for char in head if char.isdigit())
    return int(digits or 0)
