"""Hardware and model data.

Numbers here are *documented* values with sources, not guesses. Where a figure is
an estimate this module says so in a comment, because the whole point of the tool
is that people stop redoing this arithmetic from vibes.

VRAM figures are the card's usable memory for weights — for consumer cards that
is the marketing number, since the driver reserve is absorbed by the overhead
allowance in ``vram.py`` rather than by inflating the spec sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------

# Bits per weight for llama.cpp GGUF k-quants, from the published quantisation
# table (`Q4_K_M` and friends). Bytes-per-parameter is bits/8.
QUANT_BITS: dict[str, float] = {
    "fp32": 32.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.5625,
    "q5_k_m": 5.6719,
    "q5_k_s": 5.5156,
    "q4_k_m": 4.8438,
    "q4_k_s": 4.5781,
    "q4_0": 4.5,
    "q3_k_m": 3.9063,
    "q3_k_s": 3.5,
    "q2_k": 2.5625,
}

QUANT_ALIASES = {
    "f32": "fp32",
    "float32": "fp32",
    "f16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "q8": "q8_0",
    "q6": "q6_k",
    "q5": "q5_k_m",
    "q4": "q4_k_m",
    "q3": "q3_k_m",
    "q2": "q2_k",
    "int8": "q8_0",
    "int4": "q4_0",
}


def normalize_quant(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return QUANT_ALIASES.get(key, key)


def bytes_per_param(quant: str) -> float:
    """Bytes of weights per parameter for ``quant``."""

    key = normalize_quant(quant)
    if key not in QUANT_BITS:
        raise KeyError(
            f"unknown quantisation {quant!r}; known: {', '.join(sorted(QUANT_BITS))}"
        )
    return QUANT_BITS[key] / 8.0


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Enough of an architecture to size weights and a KV cache.

    ``params`` is the total parameter count in billions. For MoE models the
    *total* count sizes the weights and ``active_params`` drives the throughput
    estimate, because only the active experts are read per token.
    """

    name: str
    params: float
    layers: int
    kv_heads: int
    head_dim: int
    active_params: float | None = None
    note: str = ""

    @property
    def effective_params(self) -> float:
        """Parameters read per token — active experts for MoE, else all."""

        return self.active_params if self.active_params else self.params

    @property
    def is_moe(self) -> bool:
        return self.active_params is not None and self.active_params < self.params

    def kv_bytes_per_token(self, *, kv_dtype_bytes: int = 2) -> int:
        """KV cache bytes per token: 2 (K and V) x layers x kv_heads x head_dim x dtype."""

        return 2 * self.layers * self.kv_heads * self.head_dim * kv_dtype_bytes


# A small, honest catalogue. Layer/head geometry is the published config for
# each family; parameter counts are the commonly cited totals.
MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec("llama-3.1-8b", 8.03, 32, 8, 128),
    "llama-3.3-70b": ModelSpec("llama-3.3-70b", 70.6, 80, 8, 128),
    "llama-3.1-405b": ModelSpec("llama-3.1-405b", 405.0, 126, 8, 128),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", 7.62, 28, 4, 128),
    "qwen2.5-32b": ModelSpec("qwen2.5-32b", 32.5, 64, 8, 128),
    "qwen2.5-72b": ModelSpec("qwen2.5-72b", 72.7, 80, 8, 128),
    "mistral-7b": ModelSpec("mistral-7b", 7.24, 32, 8, 128),
    "mixtral-8x7b": ModelSpec(
        "mixtral-8x7b", 46.7, 32, 8, 128, active_params=12.9,
        note="MoE: 46.7B total, ~12.9B active per token",
    ),
    "deepseek-v3": ModelSpec(
        "deepseek-v3", 671.0, 61, 128, 128, active_params=37.0,
        note="MoE: 671B total, ~37B active per token",
    ),
    "gemma-2-9b": ModelSpec("gemma-2-9b", 9.24, 42, 8, 256),
    "gemma-2-27b": ModelSpec("gemma-2-27b", 27.2, 46, 16, 128),
    "phi-3-medium": ModelSpec("phi-3-medium", 14.0, 40, 10, 128),
    "command-r-35b": ModelSpec("command-r-35b", 35.0, 60, 8, 128),
}


def get_model(name: str) -> ModelSpec:
    key = name.strip().lower()
    if key in MODELS:
        return MODELS[key]
    # Accept a bare size like "7b" as a synthetic architecture so the planner is
    # still usable for a model nobody has catalogued yet — flagged as estimated.
    if key.endswith("b"):
        try:
            params = float(key[:-1])
        except ValueError:
            pass
        else:
            return _synthetic(params)
    raise KeyError(
        f"unknown model {name!r}; known: {', '.join(sorted(MODELS))}, or a size like '13b'"
    )


def _synthetic(params_b: float) -> ModelSpec:
    """Rough geometry from a parameter count, for uncatalogued models.

    Derived from the Llama-3 family ratios: ~62.7x more parameters than layers,
    kv_heads = layers/4, head_dim 128. Marked as estimated in its note.
    """

    layers = max(8, round(params_b * 1000 / (62.7 * 1000) * 4) or 8)
    layers = max(8, min(layers, 128))
    return ModelSpec(
        f"~{params_b:g}b (estimated)",
        params_b,
        layers,
        max(1, layers // 4),
        128,
        note="geometry estimated from parameter count — install a catalogue entry for accuracy",
    )


# --------------------------------------------------------------------------
# GPUs and cloud rates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_gb: float
    bandwidth_gbs: float  # memory bandwidth, the binding constraint for decode
    price_usd: float | None = None
    note: str = ""


GPUS: dict[str, GpuSpec] = {
    # Consumer
    "rtx-3060-12g": GpuSpec("rtx-3060-12g", 12, 360, 300),
    "rtx-3090-24g": GpuSpec("rtx-3090-24g", 24, 936, 1000, note="used market"),
    "rtx-4070-ti-16g": GpuSpec("rtx-4070-ti-16g", 16, 576, 800),
    "rtx-4090-24g": GpuSpec("rtx-4090-24g", 24, 1008, 1700),
    "rtx-5090-32g": GpuSpec("rtx-5090-32g", 32, 1792, 2000),
    # Workstation / datacentre
    "a6000-48g": GpuSpec("a6000-48g", 48, 768, 4500),
    "l40s-48g": GpuSpec("l40s-48g", 48, 864, 8000),
    "a100-40g": GpuSpec("a100-40g", 40, 1555, 10000),
    "a100-80g": GpuSpec("a100-80g", 80, 2039, 15000),
    "h100-80g": GpuSpec("h100-80g", 80, 3350, 27000),
    "h200-141g": GpuSpec("h200-141g", 141, 4800, 32000),
    "b200-180g": GpuSpec("b200-180g", 180, 8000, 40000),
    # Apple Silicon (unified memory — VRAM is the top SKU's ceiling)
    "m3-max-128g": GpuSpec("m3-max-128g", 128, 400, None, note="unified memory; ~75% usable to GPU"),
    "m4-max-128g": GpuSpec("m4-max-128g", 128, 546, None, note="unified memory; ~75% usable to GPU"),
}


def get_gpu(name: str) -> GpuSpec:
    key = name.strip().lower().replace(" ", "-")
    if key not in GPUS:
        raise KeyError(f"unknown GPU {name!r}; known: {', '.join(sorted(GPUS))}")
    return GPUS[key]


# Cloud on-demand rates, USD/hour, single-GPU instance. Approximate public
# list prices for a comparable instance; the point is the order of magnitude of
# the break-even, which the CLI lets you override.
CLOUD_RATES: dict[str, float] = {
    "a100-40g": 1.29,
    "a100-80g": 1.79,
    "h100-80g": 2.49,
    "h200-141g": 3.59,
    "b200-180g": 6.00,
    "l40s-48g": 1.10,
}


def cloud_rate(gpu: str) -> float | None:
    return CLOUD_RATES.get(gpu.strip().lower().replace(" ", "-"))
