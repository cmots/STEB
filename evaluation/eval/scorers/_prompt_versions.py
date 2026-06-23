"""Prompt-version helpers for the public STEB LLM scorers."""

from __future__ import annotations

from typing import Mapping, TypeVar


DEFAULT_PROMPT_VERSION = "v4_choice"
SUPPORTED_PROMPT_VERSIONS = (DEFAULT_PROMPT_VERSION,)

_T = TypeVar("_T")


def normalize_prompt_version(prompt_version: str | None) -> str:
    """Normalize and validate a scorer prompt version.

    The package keeps only the current automatic evaluator prompt.
    """
    version = (prompt_version or DEFAULT_PROMPT_VERSION).strip()
    if version not in SUPPORTED_PROMPT_VERSIONS:
        choices = ", ".join(SUPPORTED_PROMPT_VERSIONS)
        raise ValueError(f"Unsupported prompt_version={version!r}; choose one of: {choices}")
    return version


def select_prompt(prompt_map: Mapping[str, _T], prompt_version: str | None) -> _T:
    """Return the prompt object for ``prompt_version`` after validation."""
    version = normalize_prompt_version(prompt_version)
    return prompt_map[version]


def versioned_name(base_name: str, prompt_version: str | None) -> str:
    """Keep scorer names stable for the default public prompt."""
    normalize_prompt_version(prompt_version)
    return base_name
