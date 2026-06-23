"""Compatibility helpers for running qwen_asr with newer local vLLM builds."""

from __future__ import annotations

import importlib
import importlib.util
import os

import torch


def _patch_torch_accelerator_empty_cache() -> None:
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None or hasattr(accelerator, "empty_cache"):
        return

    if hasattr(torch.cuda, "empty_cache"):
        accelerator.empty_cache = torch.cuda.empty_cache


def _patch_functorch_config_defaults() -> None:
    config = importlib.import_module("torch._functorch.config")
    from torch.utils._config_module import Config, _ConfigEntry

    if not hasattr(config, "autograd_cache_normalize_inputs"):
        config._config["autograd_cache_normalize_inputs"] = _ConfigEntry(
            Config(default=False)
        )


def apply_startup_patch() -> None:
    _patch_torch_accelerator_empty_cache()
    _patch_functorch_config_defaults()


def _patch_flash_attn_rotary_probe() -> None:
    """Hide incompatible flash_attn from vLLM rotary probing when requested.

    The local ROCm Qwen3-ASR env can contain a flash_attn package whose Python
    files are importable but whose native extension is ABI-incompatible with the
    current torch build. vLLM only needs this probe for an optional rotary fast
    path; returning None lets it use its native rotary implementation instead.
    """
    flag = os.environ.get("QWEN_ASR_DISABLE_FLASH_ATTN_ROTARY", "0").lower()
    if flag not in {"1", "true", "yes", "on"}:
        return

    original_find_spec = importlib.util.find_spec
    if getattr(original_find_spec, "_qwen_asr_flash_attn_patch", False):
        return

    def find_spec(name, package=None):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return original_find_spec(name, package)

    find_spec._qwen_asr_flash_attn_patch = True  # type: ignore[attr-defined]
    importlib.util.find_spec = find_spec


def apply_patch() -> None:
    apply_startup_patch()
    _patch_flash_attn_rotary_probe()

    import qwen_asr.core.vllm_backend as backend
    from vllm import ModelRegistry
    from qwen_asr.core.vllm_backend import qwen3_asr as mod
    from vllm.model_executor.models.qwen3_asr import (
        Qwen3ASRForConditionalGeneration as vllm_qwen3_asr_model,
    )

    backend.Qwen3ASRForConditionalGeneration = vllm_qwen3_asr_model
    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration",
        "vllm.model_executor.models.qwen3_asr:Qwen3ASRForConditionalGeneration",
    )

    processor_cls = getattr(mod, "Qwen3ASRMultiModalProcessor", None)
    info_cls = getattr(mod, "Qwen3ASRProcessingInfo", None)

    if processor_cls is None or info_cls is None:
        return

    if not hasattr(processor_cls, "_get_data_parser"):
        return

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()
        expected_hidden_size = None
        get_expected_hidden_size = getattr(self, "_get_expected_hidden_size", None)
        if callable(get_expected_hidden_size):
            expected_hidden_size = get_expected_hidden_size()

        return mod.Qwen3ASRMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=expected_hidden_size,
        )

    info_cls.get_data_parser = get_data_parser
    delattr(processor_cls, "_get_data_parser")
    apply_startup_patch()
