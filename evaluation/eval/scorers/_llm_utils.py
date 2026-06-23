"""Shared utilities for LLM-based scorers."""

import json
import os
import sys
from typing import Any, Dict, Optional

# Ensure VLLMClient is importable
_UTILS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "core_functional_modules", "utils"
)
sys.path.insert(0, _UTILS_PATH)

from vllm_client import VLLMClient  # noqa: E402


def get_vllm_client(base_url: str, model_name: str) -> VLLMClient:
    return VLLMClient(base_url=base_url, model_name=model_name)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        return json.loads(text)
    except Exception:
        return None
