#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR模型配置
统一管理模型名称、版本等配置
"""

import os

# 模型版本，可通过环境变量覆盖
MODEL_REVISION = os.environ.get("FUNASR_MODEL_REVISION", "v2.0.5")
DEFAULT_ASR_BACKEND = os.environ.get("FUNASR_ASR_BACKEND", "paraformer_onnx")
DEFAULT_PARAFORMER_ASR_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
DEFAULT_SENSEVOICE_ASR_MODEL = "iic/SenseVoiceSmall-onnx"


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def normalize_asr_backend(backend):
    """Normalize user-facing backend aliases to stable internal names."""
    value = str(backend or DEFAULT_ASR_BACKEND).strip().lower().replace("-", "_")
    if value in ("paraformer", "paraformer_onnx", "funasr_onnx", "onnx"):
        return "paraformer_onnx"
    if value in ("sensevoice", "sensevoice_small", "sensevoice_onnx"):
        return "sensevoice_onnx"
    raise ValueError(f"不支持的 ASR 后端: {backend}")


def resolve_asr_config(config=None):
    """Resolve ASR backend/model settings from defaults, config, and env vars."""
    cfg = dict(config or {})
    backend = normalize_asr_backend(
        os.environ.get("FUNASR_ASR_BACKEND") or cfg.get("backend") or DEFAULT_ASR_BACKEND
    )

    if backend == "sensevoice_onnx":
        default_model = os.environ.get("FUNASR_SENSEVOICE_MODEL", DEFAULT_SENSEVOICE_ASR_MODEL)
        default_revision = os.environ.get("FUNASR_SENSEVOICE_REVISION") or cfg.get("model_revision") or None
        default_language = "auto"
        default_use_punc = False
        default_quantize = False
        default_auto_export = False
    else:
        default_model = os.environ.get("FUNASR_PARAFORMER_MODEL", DEFAULT_PARAFORMER_ASR_MODEL)
        default_revision = os.environ.get("FUNASR_PARAFORMER_REVISION") or cfg.get("model_revision") or MODEL_REVISION
        default_language = "zh"
        default_use_punc = True
        default_quantize = True
        default_auto_export = False

    model = os.environ.get("FUNASR_ASR_MODEL") or cfg.get("model") or cfg.get("name") or default_model
    revision = os.environ.get("FUNASR_ASR_MODEL_REVISION") or default_revision

    return {
        "backend": backend,
        "model": model,
        "model_revision": revision,
        "language": os.environ.get("FUNASR_ASR_LANGUAGE") or cfg.get("language", default_language),
        "use_itn": _parse_bool(
            os.environ.get("FUNASR_SENSEVOICE_USE_ITN", cfg.get("use_itn")),
            default=True,
        ),
        "quantize": _parse_bool(
            os.environ.get("FUNASR_ASR_QUANTIZE", cfg.get("quantize")),
            default=default_quantize,
        ),
        "auto_export_onnx": _parse_bool(
            os.environ.get("FUNASR_AUTO_EXPORT_ONNX", cfg.get("auto_export_onnx")),
            default=default_auto_export,
        ),
        "batch_size": int(cfg.get("batch_size", 1) or 1),
        "use_vad": _parse_bool(
            os.environ.get("FUNASR_USE_VAD", cfg.get("use_vad")),
            default=False,
        ),
        "use_punc": _parse_bool(
            os.environ.get("FUNASR_USE_PUNC", cfg.get("use_punc")),
            default=default_use_punc,
        ),
    }

# 模型配置（默认使用 ONNX 版本，仍可通过环境变量覆盖）
_DEFAULT_ASR_CONFIG = resolve_asr_config()
MODELS = {
    "asr": {
        "name": _DEFAULT_ASR_CONFIG["model"],
        "type": "asr",
        "backend": _DEFAULT_ASR_CONFIG["backend"],
        "revision": _DEFAULT_ASR_CONFIG["model_revision"],
    },
    "vad": {
        "name": os.environ.get(
            "FUNASR_VAD_MODEL",
            "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
        ),
        "type": "vad",
    },
    "punc": {
        "name": os.environ.get(
            "FUNASR_PUNC_MODEL",
            "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        ),
        "type": "punc",
    },
}

# 获取模型列表（用于下载脚本）
def get_models_for_download(asr_config=None):
    """返回用于下载的模型配置列表"""
    resolved_asr = resolve_asr_config(asr_config)
    models = [
        {
            "name": resolved_asr["model"],
            "type": "asr",
            "backend": resolved_asr["backend"],
            "revision": resolved_asr["model_revision"],
        }
    ]
    if resolved_asr["use_vad"]:
        models.append({
            "name": MODELS["vad"]["name"],
            "type": "vad",
            "revision": MODEL_REVISION,
        })
    if resolved_asr["use_punc"]:
        models.append({
            "name": MODELS["punc"]["name"],
            "type": "punc",
            "revision": MODEL_REVISION,
        })
    return models
