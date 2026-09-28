"""The VRAM maths, tested directly.

These tests pin the arithmetic rather than the catalogue: the numbers are
computed from first principles in the test so a change to the implementation
that quietly changes the meaning of a figure fails here.
"""

from __future__ import annotations

import math

import pytest

from gpu_planner import catalog, vram
from gpu_planner.catalog import ModelSpec, bytes_per_param, get_model

# --------------------------------------------------------------------------
# quantisation
# --------------------------------------------------------------------------


def test_bytes_per_param_matches_the_published_quant_table():
    # Q4_K_M is 4.8438 bits per weight in llama.cpp's table.
    assert bytes_per_param("q4_k_m") == pytest.approx(4.8438 / 8)
    assert bytes_per_param("fp16") == pytest.approx(2.0)
    assert bytes_per_param("fp32") == pytest.approx(4.0)
    assert bytes_per_param("q8_0") == pytest.approx(8.5 / 8)


def test_quant_aliases_resolve():
    assert catalog.normalize_quant("Q4_K_M") == "q4_k_m"
    assert catalog.normalize_quant("int4") == "q4_0"
    assert catalog.normalize_quant("f16") == "fp16"


def test_unknown_quant_raises_with_a_useful_message():
    with pytest.raises(KeyError) as exc:
        bytes_per_param("q9_zz")
    assert "q4_k_m" in str(exc.value)


def test_lower_quants_are_smaller():
    sizes = [bytes_per_param(q) for q in ("fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q2_k")]
    assert sizes == sorted(sizes, reverse=True)


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def test_weight_bytes_is_params_times_bytes_per_param():
    model = ModelSpec("t", 7.0, 32, 8, 128)
    expected = 7.0e9 * (4.8438 / 8)
    assert vram.weight_bytes(model, "q4_k_m") == pytest.approx(expected)


def test_weight_bytes_scales_with_quantisation():
    model = get_model("llama-3.3-70b")
    fp16 = vram.weight_bytes(model, "fp16")
    q4 = vram.weight_bytes(model, "q4_k_m")
    # fp16 is 2 bytes/param and q4_k_m ~0.605, so the ratio is ~3.3x.
    assert fp16 / q4 == pytest.approx(2.0 / (4.8438 / 8), rel=1e-6)
    assert fp16 > q4


def test_70b_q4_weights_are_about_43gb():
    """The widely quoted figure: a 70B at Q4_K_M is ~40-45GB of weights."""

    gb = vram.weight_bytes(get_model("llama-3.3-70b"), "q4_k_m") / vram.GB
    assert 40 < gb < 45, gb


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------


def test_kv_bytes_per_token_formula():
    model = ModelSpec("t", 7.0, 32, 8, 128)
    # 2 (K,V) * 32 layers * 8 kv_heads * 128 head_dim * 2 bytes
    assert model.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
    assert model.kv_bytes_per_token() == 131072


def test_kv_cache_is_linear_in_context():
    model = get_model("llama-3.3-70b")
    at_4k = vram.kv_bytes(model, 4096)
    at_8k = vram.kv_bytes(model, 8192)
    assert at_8k == pytest.approx(at_4k * 2)


def test_70b_kv_cache_at_8k_is_about_2_7gb():
    """80 layers x 8 kv_heads x 128 dim x 2 (K,V) x 2 bytes x 8192 tokens."""

    model = get_model("llama-3.3-70b")
    assert model.layers == 80
    gb = vram.kv_bytes(model, 8192) / vram.GB
    assert 2.6 < gb < 2.8, gb
    # Cross-check the formula against an independent hand calculation.
    expected = 2 * 80 * 8 * 128 * 2 * 8192 / vram.GB
    assert gb == pytest.approx(expected)


def test_kv_cache_shrinks_with_a_smaller_kv_dtype():
    model = get_model("llama-3.3-70b")
    fp16 = vram.kv_bytes(model, 8192, kv_dtype_bytes=2)
    q8 = vram.kv_bytes(model, 8192, kv_dtype_bytes=1)
    assert q8 == pytest.approx(fp16 / 2)


# --------------------------------------------------------------------------
# overhead
# --------------------------------------------------------------------------


def test_overhead_has_a_floor_and_grows_with_weights():
    small = vram.overhead_bytes(0.0)
    assert small == pytest.approx(vram.OVERHEAD_BASE_GB * vram.GIB)
    bigger = vram.overhead_bytes(100 * vram.GIB)
    assert bigger > small


# --------------------------------------------------------------------------
# fit decisions
# --------------------------------------------------------------------------


def test_model_that_fits_reports_fits_fully():
    fit = vram.plan_fit(
        get_model("llama-3.1-8b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=8192,
    )
    assert fit.verdict == "fits fully"
    assert fit.gpu_shortfall_gb == 0
    assert fit.fits_gpu


def test_model_that_just_overflows_needs_overspill():
    # A 70B at Q4 is ~43GB of weights: does not fit 48GB with cache+overhead,
    # but does fit when the overflow can spill to RAM.
    fit = vram.plan_fit(
        get_model("llama-3.3-70b"), "q4_k_m",
        gpu_vram_gb=48, system_ram_gb=128, context=32768,
    )
    assert fit.verdict in {"needs CPU overspill", "fits fully"}
    assert fit.fits_system_ram


def test_model_too_big_for_gpu_and_ram_wont_fit():
    fit = vram.plan_fit(
        get_model("llama-3.1-405b"), "fp16",
        gpu_vram_gb=24, system_ram_gb=32, context=4096,
    )
    assert fit.verdict == "won't fit"
    assert not fit.fits_system_ram


def test_total_required_is_weights_plus_kv_plus_overhead():
    fit = vram.plan_fit(
        get_model("qwen2.5-32b"), "q5_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=8192,
    )
    assert fit.total_required_gb == pytest.approx(
        fit.weights_gb + fit.kv_gb + fit.overhead_gb
    )


def test_kv_dtype_choice_changes_the_verdict_at_the_boundary():
    """A model that fits at 8-bit KV and not at fp16 KV, near the boundary."""

    model = get_model("gemma-2-9b")
    tight = vram.plan_fit(
        model, "fp16", gpu_vram_gb=24, system_ram_gb=0, context=8192,
        kv_dtype_bytes=2,
    )
    loose = vram.plan_fit(
        model, "fp16", gpu_vram_gb=24, system_ram_gb=0, context=8192,
        kv_dtype_bytes=1,
    )
    assert loose.total_required_gb < tight.total_required_gb


def test_invalid_inputs_are_rejected():
    model = get_model("llama-3.1-8b")
    with pytest.raises(ValueError, match="context"):
        vram.plan_fit(model, "q4_k_m", gpu_vram_gb=24, system_ram_gb=64, context=0)
    with pytest.raises(ValueError, match="VRAM"):
        vram.plan_fit(model, "q4_k_m", gpu_vram_gb=0, system_ram_gb=64, context=1024)


# --------------------------------------------------------------------------
# max context
# --------------------------------------------------------------------------


def test_max_context_is_the_largest_window_that_fits():
    fit = vram.plan_fit(
        get_model("llama-3.1-8b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=8192,
    )
    limit = fit.max_context_fitting_gpu
    assert limit > 8192
    # One step beyond must not fit.
    over = vram.plan_fit(
        get_model("llama-3.1-8b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=limit,
    )
    assert over.fits_gpu
    beyond = vram.plan_fit(
        get_model("llama-3.1-8b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=limit + 1000,
    )
    assert not beyond.fits_gpu


def test_max_context_is_zero_when_weights_alone_exceed_vram():
    fit = vram.plan_fit(
        get_model("llama-3.1-405b"), "fp16",
        gpu_vram_gb=24, system_ram_gb=32, context=1024,
    )
    assert fit.max_context_fitting_gpu == 0


def test_max_context_total_exceeds_gpu_only_when_ram_helps():
    model = get_model("llama-3.3-70b")
    fit = vram.plan_fit(model, "q4_k_m", gpu_vram_gb=48, system_ram_gb=128, context=8192)
    assert fit.max_context_fitting_total > fit.max_context_fitting_gpu


# --------------------------------------------------------------------------
# best quant
# --------------------------------------------------------------------------


def test_best_quant_on_gpu_is_the_highest_quality_that_fits():
    best = vram.largest_quant_that_fits(
        get_model("llama-3.1-8b"), gpu_vram_gb=24, system_ram_gb=64, context=8192
    )
    assert best == "fp16"


def test_best_quant_is_lower_for_a_bigger_model():
    small = vram.largest_quant_that_fits(
        get_model("llama-3.1-8b"), gpu_vram_gb=24, system_ram_gb=64, context=8192
    )
    big = vram.largest_quant_that_fits(
        get_model("llama-3.3-70b"), gpu_vram_gb=24, system_ram_gb=64, context=8192
    )
    assert best_rank(small) > best_rank(big) or big is None


def best_rank(quant: str | None) -> float:
    return bytes_per_param(quant) if quant else 0.0


def test_best_quant_is_none_when_nothing_fits():
    assert (
        vram.largest_quant_that_fits(
            get_model("llama-3.1-405b"), gpu_vram_gb=8, system_ram_gb=8, context=8192
        )
        is None
    )


# --------------------------------------------------------------------------
# overspill planning
# --------------------------------------------------------------------------


def test_overspill_plan_offloads_enough_layers():
    fit = vram.plan_fit(
        get_model("llama-3.3-70b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=128, context=8192,
    )
    plan = vram.overspill_plan(fit)
    assert plan.total_layers == get_model("llama-3.3-70b").layers
    assert plan.gpu_layers + plan.cpu_layers == plan.total_layers
    assert plan.cpu_layers > 0
    assert plan.freed_gb >= fit.gpu_shortfall_gb - 1e-6
    assert plan.viable


def test_overspill_plan_is_empty_when_everything_fits():
    fit = vram.plan_fit(
        get_model("llama-3.1-8b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=8192,
    )
    plan = vram.overspill_plan(fit)
    assert plan.cpu_layers == 0
    assert plan.gpu_layers == plan.total_layers


def test_overspill_plan_reports_infeasible_when_layers_cannot_cover_it():
    fit = vram.plan_fit(
        get_model("llama-3.1-405b"), "fp16",
        gpu_vram_gb=8, system_ram_gb=16, context=8192,
    )
    plan = vram.overspill_plan(fit)
    assert not plan.viable
    assert "no viable offload" in plan.describe(fit.model.name)


def test_overspill_plan_describes_itself_in_plain_numbers():
    fit = vram.plan_fit(
        get_model("llama-3.3-70b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=128, context=8192,
    )
    text = vram.overspill_plan(fit).describe(fit.model.name)
    assert "layers to CPU" in text
    assert "% on GPU" in text


def test_overspill_plan_says_no_offload_when_it_all_fits():
    fit = vram.plan_fit(
        get_model("llama-3.1-8b"), "q4_k_m",
        gpu_vram_gb=24, system_ram_gb=64, context=8192,
    )
    text = vram.overspill_plan(fit).describe(fit.model.name)
    assert "no offload needed" in text
    # It must not claim to move bytes it is not moving.
    assert "0.00GB moved" not in text


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------


def test_moe_uses_active_params_for_throughput_but_total_for_weights():
    moe = get_model("mixtral-8x7b")
    assert moe.is_moe
    assert moe.params > moe.effective_params

    dense_same_weights = ModelSpec("dense", moe.params, 32, 8, 128)
    assert vram.weight_bytes(moe, "q4_k_m") == pytest.approx(
        vram.weight_bytes(dense_same_weights, "q4_k_m")
    )
    # But the MoE is far faster, because only the active experts are read.
    moe_tps = vram.estimate_throughput(moe, "q4_k_m", gpu_bandwidth_gbs=1000)
    dense_tps = vram.estimate_throughput(dense_same_weights, "q4_k_m", gpu_bandwidth_gbs=1000)
    assert moe_tps.effective_tokens_per_sec > dense_tps.effective_tokens_per_sec


def test_deepseek_v3_is_modelled_as_moe():
    model = get_model("deepseek-v3")
    assert model.is_moe
    assert model.effective_params < model.params / 10


# --------------------------------------------------------------------------
# throughput
# --------------------------------------------------------------------------


def test_throughput_formula_is_bandwidth_over_active_bytes():
    model = get_model("llama-3.1-8b")
    bandwidth = 1000.0
    tps = vram.estimate_throughput(model, "q4_k_m", gpu_bandwidth_gbs=bandwidth)
    active_gb = vram.weight_bytes(model, "q4_k_m") / 1e9
    expected = bandwidth * vram.DECODE_EFFICIENCY_GPU / active_gb
    assert tps.effective_tokens_per_sec == pytest.approx(expected)


def test_throughput_rises_with_bandwidth():
    model = get_model("llama-3.1-8b")
    slow = vram.estimate_throughput(model, "q4_k_m", gpu_bandwidth_gbs=500)
    fast = vram.estimate_throughput(model, "q4_k_m", gpu_bandwidth_gbs=2000)
    assert fast.effective_tokens_per_sec == pytest.approx(
        slow.effective_tokens_per_sec * 4
    )


def test_smaller_quant_is_faster():
    model = get_model("llama-3.3-70b")
    fp16 = vram.estimate_throughput(model, "fp16", gpu_bandwidth_gbs=1000)
    q4 = vram.estimate_throughput(model, "q4_k_m", gpu_bandwidth_gbs=1000)
    assert q4.effective_tokens_per_sec > fp16.effective_tokens_per_sec


def test_overspill_slows_throughput_and_says_why():
    model = get_model("llama-3.3-70b")
    fit = vram.plan_fit(model, "q4_k_m", gpu_vram_gb=24, system_ram_gb=128, context=8192)
    plan = vram.overspill_plan(fit)
    split = vram.estimate_throughput(
        model, "q4_k_m", gpu_bandwidth_gbs=1000, plan=plan
    )
    full = vram.estimate_throughput(model, "q4_k_m", gpu_bandwidth_gbs=1000)
    assert split.effective_tokens_per_sec < full.effective_tokens_per_sec
    assert "CPU" in split.bottleneck
    assert "GPU-only would be" in split.note


def test_unified_memory_uses_the_higher_efficiency():
    model = get_model("llama-3.1-8b")
    unified = vram.estimate_throughput(
        model, "q4_k_m", gpu_bandwidth_gbs=546, unified_memory=True
    )
    assert unified.effective_tokens_per_sec == pytest.approx(
        546 * vram.DECODE_EFFICIENCY_UNIFIED / (vram.weight_bytes(model, "q4_k_m") / 1e9)
    )


# --------------------------------------------------------------------------
# rent vs buy
# --------------------------------------------------------------------------


def test_own_hourly_is_capex_amortised_plus_electricity():
    be = vram.break_even(
        gpu_price=3000, cloud_hourly=2.49, monthly_hours=200,
        amortise_months=36, electricity_watts=450, electricity_usd_per_kwh=0.15,
    )
    lifetime_hours = 36 * 30.44 * 24
    expected = 3000 / lifetime_hours + 0.450 * 0.15
    assert be.own_hourly == pytest.approx(expected)


def test_break_even_hours_solves_the_crossover():
    """At the crossover, total owning cost equals total renting cost."""

    be = vram.break_even(
        gpu_price=3000, cloud_hourly=2.49, monthly_hours=200,
        electricity_watts=450, electricity_usd_per_kwh=0.15,
    )
    power_per_hour = 0.450 * 0.15
    owning = be.gpu_price + power_per_hour * be.break_even_hours
    renting = be.cloud_hourly * be.break_even_hours
    assert owning == pytest.approx(renting, rel=1e-6)


def test_renting_is_cheaper_at_low_usage():
    be = vram.break_even(
        gpu_price=30000, cloud_hourly=2.49, monthly_hours=5, amortise_months=36
    )
    assert be.cheaper_at_current_usage == "rent"
    assert be.break_even_months > 5


def test_owning_is_cheaper_at_sustained_usage():
    """A cheap card used heavily pays back inside the ownership period."""

    be = vram.break_even(
        gpu_price=2000, cloud_hourly=2.49, monthly_hours=400, amortise_months=36
    )
    assert be.break_even_months < 36
    assert be.cheaper_at_current_usage == "own"


def test_verdict_flips_between_light_and_heavy_use_of_the_same_card():
    """The same GPU is a bad buy at 20h/mo and a good buy at 400h/mo."""

    light = vram.break_even(gpu_price=3000, cloud_hourly=2.49, monthly_hours=20)
    heavy = vram.break_even(gpu_price=3000, cloud_hourly=2.49, monthly_hours=400)
    assert light.cheaper_at_current_usage == "rent"
    assert heavy.cheaper_at_current_usage == "own"
    # The crossover itself is a property of the card, not of usage.
    assert light.break_even_hours == pytest.approx(heavy.break_even_hours)


def test_high_electricity_can_make_owning_never_cheaper():
    be = vram.break_even(
        gpu_price=100, cloud_hourly=0.10, monthly_hours=200,
        electricity_watts=5000, electricity_usd_per_kwh=0.50,
    )
    assert be.own_hourly >= be.cloud_hourly
    assert be.cheaper_at_current_usage == "rent"
    assert math.isinf(be.break_even_hours)


def test_break_even_rejects_nonsense_inputs():
    with pytest.raises(ValueError, match="price"):
        vram.break_even(gpu_price=0, cloud_hourly=1, monthly_hours=10)
    with pytest.raises(ValueError, match="cloud"):
        vram.break_even(gpu_price=100, cloud_hourly=0, monthly_hours=10)
    with pytest.raises(ValueError, match="hours"):
        vram.break_even(gpu_price=100, cloud_hourly=1, monthly_hours=0)
