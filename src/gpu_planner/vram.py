"""VRAM arithmetic.

Three things have to fit: weights, the KV cache, and the runtime's own overhead.
Getting only the first one right is why people think a 70B fits on two 24GB
cards and then watch it OOM at 4k context.

Every function here is pure so the maths can be tested directly, which is what
the work order asks for.

One unit convention runs through the whole module: **memory budgets are
gigabytes (1e9)**, matching how GPUs are sold ("24GB card"), while the KV cache
uses GiB for reporting because tooling quotes it that way. Mixing the two is the
classic source of a wrong answer here, so the arithmetic below is explicit about
which it is doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import ModelSpec, bytes_per_param

GIB = 1024**3
GB = 1e9

# llama.cpp / CUDA context, cuBLAS workspaces, fragmentation. Empirically
# 0.5-1.5GB for small models and more for big ones; scaled by weight size.
OVERHEAD_BASE_GB = 0.60
OVERHEAD_FRACTION = 0.02  # of weight bytes


def weight_bytes(model: ModelSpec, quant: str) -> float:
    """Bytes of quantised weights."""

    return model.params * 1e9 * bytes_per_param(quant)


def kv_bytes(model: ModelSpec, context: int, *, kv_dtype_bytes: int = 2) -> float:
    """KV cache bytes for a full ``context`` window."""

    return model.kv_bytes_per_token(kv_dtype_bytes=kv_dtype_bytes) * context


def overhead_bytes(weights: float) -> float:
    """Runtime overhead: a fixed floor plus a fraction of the weights."""

    return (OVERHEAD_BASE_GB * GIB) + weights * OVERHEAD_FRACTION


@dataclass
class Fit:
    """The result of trying to place a model on some memory."""

    model: ModelSpec
    quant: str
    context: int
    weights_gb: float
    kv_gb: float
    overhead_gb: float
    gpu_vram_gb: float
    system_ram_gb: float

    @property
    def total_required_gb(self) -> float:
        return self.weights_gb + self.kv_gb + self.overhead_gb

    @property
    def fits_gpu(self) -> bool:
        return self.total_required_gb <= self.gpu_vram_gb

    @property
    def gpu_shortfall_gb(self) -> float:
        """How much has to move off the GPU. 0 when it fits."""

        return max(0.0, self.total_required_gb - self.gpu_vram_gb)

    @property
    def fits_system_ram(self) -> bool:
        """Whether the overflow has somewhere to go on the CPU side."""

        return self.gpu_shortfall_gb <= self.system_ram_gb

    @property
    def verdict(self) -> str:
        if self.fits_gpu:
            return "fits fully"
        if not self.fits_system_ram:
            return "won't fit"
        return "needs CPU overspill"

    @property
    def max_context_fitting_gpu(self) -> int:
        """Largest context that keeps everything on the GPU (0 if never)."""

        return _max_context(
            self.model, self.quant, self.gpu_vram_gb, self.system_ram_gb, gpu_only=True
        )

    @property
    def max_context_fitting_total(self) -> int:
        """Largest context that fits in GPU + RAM together."""

        return _max_context(
            self.model, self.quant, self.gpu_vram_gb, self.system_ram_gb, gpu_only=False
        )


def _max_context(
    model: ModelSpec, quant: str, vram_gb: float, ram_gb: float, *, gpu_only: bool
) -> int:
    """Largest context that fits, solved directly.

    Context growth is linear in KV bytes, so the closed form is exact — no need
    to search. The budget is decimal GB, matching how the card is sold.
    """

    budget = (vram_gb if gpu_only else vram_gb + ram_gb) * GB
    weights = weight_bytes(model, quant)
    fixed = weights + overhead_bytes(weights)
    per_token = model.kv_bytes_per_token()

    if fixed >= budget or per_token <= 0:
        return 0
    return int((budget - fixed) // per_token)


def plan_fit(
    model: ModelSpec,
    quant: str,
    *,
    gpu_vram_gb: float,
    system_ram_gb: float,
    context: int,
    kv_dtype_bytes: int = 2,
) -> Fit:
    """Size a model for a given memory budget and context.

    All reported figures are decimal GB, the unit GPUs are sold in.
    """

    if context < 1:
        raise ValueError("context must be at least 1 token")
    if gpu_vram_gb <= 0:
        raise ValueError("gpu VRAM must be positive")

    weights = weight_bytes(model, quant)
    return Fit(
        model=model,
        quant=quant,
        context=context,
        weights_gb=weights / GB,
        kv_gb=kv_bytes(model, context, kv_dtype_bytes=kv_dtype_bytes) / GB,
        overhead_gb=overhead_bytes(weights) / GB,
        gpu_vram_gb=gpu_vram_gb,
        system_ram_gb=system_ram_gb,
    )


def largest_quant_that_fits(
    model: ModelSpec, *, gpu_vram_gb: float, system_ram_gb: float, context: int
) -> str | None:
    """Highest-quality quant that fits entirely on the GPU, else None.

    Ordered by bytes-per-parameter descending, so the first that fits is the
    best quality.
    """

    from .catalog import QUANT_BITS

    ordered = sorted(QUANT_BITS, key=lambda q: -bytes_per_param(q))
    for quant in ordered:
        if plan_fit(
            model,
            quant,
            gpu_vram_gb=gpu_vram_gb,
            system_ram_gb=system_ram_gb,
            context=context,
        ).fits_gpu:
            return quant
    return None


@dataclass
class OverspillPlan:
    """A concrete layer-offload plan for a model that does not fit on the GPU."""

    gpu_layers: int
    cpu_layers: int
    total_layers: int
    freed_gb: float
    still_over_gb: float

    @property
    def viable(self) -> bool:
        return self.gpu_layers > 0 and self.still_over_gb <= 0

    def describe(self, model_name: str) -> str:
        if not self.viable:
            return "no viable offload plan — reduce quant, shorten context, or add VRAM"
        if self.cpu_layers == 0:
            return (
                f"no offload needed — all {self.total_layers} layers fit on the GPU"
            )
        percent = 100.0 * self.gpu_layers / max(self.total_layers, 1)
        return (
            f"offload {self.cpu_layers}/{self.total_layers} layers to CPU "
            f"({percent:.0f}% on GPU, {self.gpu_layers} layers); "
            f"{self.freed_gb:.2f}GB moved off the GPU"
        )


def overspill_plan(fit: Fit, *, context_share: float = 1.0) -> OverspillPlan:
    """Work out how many layers must move to CPU for the rest to fit.

    Layer sizes are assumed uniform, which is what llama.cpp's ``--n-gpu-layers``
    assumes in practice. The KV cache stays with whichever device holds the
    layer, so offloading a layer frees its weights plus its slice of the cache.
    """

    model = fit.model
    total = model.layers
    if total <= 0:
        return OverspillPlan(0, 0, 0, 0.0, fit.total_required_gb)

    per_layer_weights = fit.weights_gb / total
    per_layer_kv = fit.kv_gb / total * context_share
    per_layer = per_layer_weights + per_layer_kv
    if per_layer <= 0:
        return OverspillPlan(total, 0, total, 0.0, 0.0)

    needed = fit.gpu_shortfall_gb
    layers_off = min(total, int(needed / per_layer) + (1 if needed % per_layer else 0))
    gpu_layers = total - layers_off
    freed = per_layer * layers_off
    return OverspillPlan(
        gpu_layers=gpu_layers,
        cpu_layers=layers_off,
        total_layers=total,
        freed_gb=freed,
        still_over_gb=max(0.0, needed - freed),
    )


# --------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------

# Decode is memory-bandwidth bound: every token requires reading the active
# weights once. Real efficiency is well below the theoretical peak because of
# kernel launch overhead, attention cost, and KV reads.
DECODE_EFFICIENCY_GPU = 0.55
# Apple Silicon is more efficient per byte for this workload (unified memory, no
# PCIe copy) — the community reports better-than-theoretical on small models.
DECODE_EFFICIENCY_UNIFIED = 0.70
# When layers are split, the CPU-side layers run on system RAM bandwidth and the
# pipeline is limited by the slowest stage plus transfer overhead.
CPU_BANDWIDTH_GBS = 100.0  # dual-channel DDR5-6000, realistic


@dataclass
class Throughput:
    model: ModelSpec
    quant: str
    gpu_tokens_per_sec: float
    effective_tokens_per_sec: float
    bottleneck: str
    note: str = ""


def estimate_throughput(
    model: ModelSpec,
    quant: str,
    *,
    gpu_bandwidth_gbs: float,
    plan: OverspillPlan | None = None,
    unified_memory: bool = False,
    batch_parallel: bool = False,
    concurrency: int = 1,
) -> Throughput:
    """Estimate decode tokens/sec.

    Single-stream decode reads the weights once per token, so
    ``tokens/s = bandwidth / active_weight_bytes * efficiency``.
    """

    active_bytes = model.effective_params * 1e9 * bytes_per_param(quant)
    if active_bytes <= 0:
        raise ValueError("model has no parameters")

    if unified_memory:
        efficiency = DECODE_EFFICIENCY_UNIFIED
        raw = gpu_bandwidth_gbs * efficiency
        bottleneck = "unified memory bandwidth"
    else:
        efficiency = DECODE_EFFICIENCY_GPU
        raw = gpu_bandwidth_gbs * efficiency
        bottleneck = "GPU memory bandwidth"

    gpu_tps = raw / (active_bytes / 1e9)

    if plan is None or not plan.viable or plan.cpu_layers == 0:
        return Throughput(
            model=model,
            quant=quant,
            gpu_tokens_per_sec=gpu_tps,
            effective_tokens_per_sec=gpu_tps,
            bottleneck=bottleneck,
            note="all layers on GPU" + (f" (MoE: {model.effective_params:g}B active)" if model.is_moe else ""),
        )

    # Split execution: GPU layers run at GPU speed, CPU layers at RAM speed, and
    # the two stages serialise per token, so the slower stage dominates.
    cpu_bytes = active_bytes * (plan.cpu_layers / plan.total_layers)
    gpu_bytes = active_bytes * (plan.gpu_layers / plan.total_layers)

    gpu_stage = 1.0 / ((gpu_bandwidth_gbs * efficiency) / (gpu_bytes / 1e9))
    cpu_stage = 1.0 / ((CPU_BANDWIDTH_GBS * 0.6) / (cpu_bytes / 1e9)) if cpu_bytes > 0 else 0.0
    # Per-token latency is the sum of the two stages; a small transfer penalty
    # accounts for activations crossing PCIe each token.
    transfer = 0.0002 * plan.cpu_layers
    per_token = gpu_stage + cpu_stage + transfer
    effective = 1.0 / per_token if per_token > 0 else 0.0

    return Throughput(
        model=model,
        quant=quant,
        gpu_tokens_per_sec=gpu_tps,
        effective_tokens_per_sec=effective,
        bottleneck="CPU/RAM stage (layers offloaded)",
        note=f"{plan.cpu_layers}/{plan.total_layers} layers on CPU; "
        f"GPU-only would be {gpu_tps:.1f} tok/s",
    )


# --------------------------------------------------------------------------
# Rent vs buy
# --------------------------------------------------------------------------


@dataclass
class BreakEven:
    gpu_price: float
    monthly_hours: float
    cloud_hourly: float
    own_hourly: float
    break_even_hours: float
    break_even_months: float
    cheaper_at_current_usage: str

    def describe(self) -> str:
        if math.isinf(self.break_even_hours):
            return (
                f"own: ${self.own_hourly:.2f}/h amortised | rent: ${self.cloud_hourly:.2f}/h | "
                f"never pays back — electricity alone (${self.own_hourly - self.gpu_price / (36 * 30.44 * 24):.2f}/h) "
                f"leaves owning no cheaper per hour, so renting wins at any usage"
            )
        return (
            f"own: ${self.own_hourly:.2f}/h amortised | rent: ${self.cloud_hourly:.2f}/h | "
            f"break-even at {self.break_even_hours:,.0f}h "
            f"({self.break_even_months:.1f} months at {self.monthly_hours:,.0f}h/mo) "
            f"-> {self.cheaper_at_current_usage} is cheaper now"
        )


def break_even(
    *,
    gpu_price: float,
    cloud_hourly: float,
    monthly_hours: float,
    amortise_months: int = 36,
    electricity_watts: float = 450.0,
    electricity_usd_per_kwh: float = 0.15,
    cloud_storage_monthly: float = 0.0,
) -> BreakEven:
    """Compare owning a GPU against renting the same class by the hour.

    Totals, not rates — that is what people actually compare. Over ``H`` hours:

        owning  = gpu_price + electricity * H
        renting = cloud_hourly * H

    Setting them equal gives the crossover ``H = gpu_price / (cloud - electricity)``.
    ``own_hourly`` is reported as an amortised rate for legibility, but the
    break-even itself uses totals, so a machine that is idle most of the month
    is correctly shown to favour renting.
    """

    if gpu_price <= 0:
        raise ValueError("gpu price must be positive")
    if cloud_hourly <= 0:
        raise ValueError("cloud hourly rate must be positive")
    if monthly_hours <= 0:
        raise ValueError("monthly hours must be positive")
    if amortise_months <= 0:
        raise ValueError("amortise months must be positive")

    lifetime_hours = amortise_months * 30.44 * 24
    capex_per_hour = gpu_price / lifetime_hours
    power_per_hour = electricity_watts / 1000.0 * electricity_usd_per_kwh
    own_hourly = capex_per_hour + power_per_hour

    # If even the electricity rate exceeds the cloud rate, owning never pays
    # back: the fixed cost can never be recovered.
    if power_per_hour >= cloud_hourly:
        return BreakEven(
            gpu_price=gpu_price,
            monthly_hours=monthly_hours,
            cloud_hourly=cloud_hourly,
            own_hourly=own_hourly,
            break_even_hours=float("inf"),
            break_even_months=float("inf"),
            cheaper_at_current_usage="rent",
        )

    break_even_hours = (gpu_price + cloud_storage_monthly) / (
        cloud_hourly - power_per_hour
    )
    break_even_months = break_even_hours / monthly_hours

    # Owning is the better deal when the crossover lands inside the ownership
    # period. Comparing monthly_hours against break_even_hours would be
    # dimensionally wrong (hours/month vs hours).
    cheaper = "own" if break_even_months <= amortise_months else "rent"
    return BreakEven(
        gpu_price=gpu_price,
        monthly_hours=monthly_hours,
        cloud_hourly=cloud_hourly,
        own_hourly=own_hourly,
        break_even_hours=break_even_hours,
        break_even_months=break_even_months,
        cheaper_at_current_usage=cheaper,
    )
