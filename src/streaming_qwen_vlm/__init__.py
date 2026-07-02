"""Streaming Qwen2.5-VL-3B VLM context-feature module + Stage-2 VLA.

Public names are imported lazily (PEP 562): light CLIs like
``python -m streaming_qwen_vlm.normalize`` must not pay the torch/transformers/flash_attn import
cost (minutes on a cold network filesystem) just because the package __init__ ran.
"""

import importlib

# name -> submodule that defines it (imported on first attribute access)
_LAZY = {
    "StreamingQwenVLM": ".model",
    "VLMConfig": ".config",
    "VLMOutput": ".outputs",
    "load_backbone": ".backbone",
    "resolve_dtype": ".backbone",
    "run_language_early_exit": ".early_exit",
    "RollingFrameBuffer": ".frame_buffer",
    "PairVisionCache": ".vision_cache",
    "StateProjector": ".state_encoder",
    "PromptTemplate": ".prompt_builder",
    "ActTemplate": ".prompt_builder",
    "build_prompt": ".prompt_builder",
    "build_act_template": ".prompt_builder",
    "TOKEN_TYPE_TEXT": ".outputs",
    "TOKEN_TYPE_VIDEO": ".outputs",
    "TOKEN_TYPE_STATE": ".outputs",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache so __getattr__ runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
