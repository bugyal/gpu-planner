"""CLI behaviour: acceptance criteria (1) inputs and (2) outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_planner import cli

EXAMPLES = Path(__file__).parent.parent / "examples"


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


BASE = ("--gpu", "rtx-4090-24g", "--ram", "64")


# --------------------------------------------------------------------------
# (1) the CLI accepts the documented inputs
# --------------------------------------------------------------------------


def test_model_and_quant_flags_are_accepted(capsys):
    code, out, _ = run_cli(capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q4_k_m")
    assert code == cli.EXIT_OK
    assert "llama-3.1-8b" in out
    assert "q4_k_m" in out


def test_inventory_file_is_accepted(capsys):
    code, out, _ = run_cli(
        capsys, "--inventory", str(EXAMPLES / "my-rig.toml"), "--model", "mistral-7b"
    )
    assert code == cli.EXIT_OK
    assert "rtx-3090-24g" in out
    assert "936" in out


def test_context_flag_changes_the_answer(capsys):
    _, small, _ = run_cli(capsys, *BASE, "--model", "llama-3.3-70b", "--context", "4096")
    _, large, _ = run_cli(capsys, *BASE, "--model", "llama-3.3-70b", "--context", "262144")
    # A much longer context needs at least as much memory.
    assert "won't fit" in large or "overspill" in large
    assert small != large


def test_multiple_models_and_quants_are_rendered(capsys):
    code, out, _ = run_cli(
        capsys, *BASE,
        "--model", "llama-3.1-8b", "--model", "qwen2.5-7b",
        "--quant", "q4_k_m", "--quant", "q8_0",
    )
    assert code == cli.EXIT_OK
    assert "llama-3.1-8b" in out and "qwen2.5-7b" in out
    assert "q4_k_m" in out and "q8_0" in out


def test_size_shorthand_works_for_uncatalogued_models(capsys):
    code, out, _ = run_cli(capsys, *BASE, "--model", "13b", "--quant", "q4_k_m")
    assert code == cli.EXIT_OK
    assert "estimated" in out  # flagged as an estimate, not presented as fact


# --------------------------------------------------------------------------
# (2) the three required outputs
# --------------------------------------------------------------------------


def test_feasibility_table_has_a_verdict_per_quant(capsys):
    _, out, _ = run_cli(capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q4_k_m", "--quant", "fp16")
    assert "verdict" in out
    assert "fits fully" in out
    assert "weights" in out and "kv" in out and "total" in out


def test_overspill_plan_is_printed(capsys):
    _, out, _ = run_cli(capsys, *BASE, "--model", "llama-3.3-70b", "--quant", "q4_k_m")
    assert "overspill plan" in out
    assert "layers to CPU" in out or "no offload needed" in out
    assert "max context fully on GPU" in out


def test_overspill_plan_offloads_for_a_70b_on_24gb(capsys):
    _, out, _ = run_cli(capsys, *BASE, "--model", "llama-3.3-70b", "--quant", "q4_k_m")
    assert "needs CPU overspill" in out
    assert "offload" in out
    # The plan must move a real number of layers, not zero.
    assert "offload 0/" not in out


def test_break_even_sheet_appears_with_rent(capsys):
    _, out, _ = run_cli(
        capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q4_k_m",
        "--rent", "--hours", "300",
    )
    assert "rent vs buy" in out
    assert "break-even at" in out
    assert "/Mtok" in out


def test_rent_sheet_states_when_it_substitutes_a_proxy_rate(capsys):
    _, out, _ = run_cli(
        capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q4_k_m", "--rent"
    )
    assert "proxy:" in out
    assert "no consumer cloud rate" in out


def test_throughput_is_reported_with_its_bottleneck(capsys):
    _, out, _ = run_cli(capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q4_k_m")
    assert "tok/s" in out
    assert "bottleneck" in out


# --------------------------------------------------------------------------
# machine-readable output
# --------------------------------------------------------------------------


def test_json_output_is_machine_readable(capsys):
    code, out, _ = run_cli(
        capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q4_k_m", "--json"
    )
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["inventory"]["gpu"] == "rtx-4090-24g"
    assert payload["inventory"]["vram_gb"] == 24
    assert payload["context"] == 8192
    q = payload["models"][0]["quants"][0]
    assert {"verdict", "weights_gb", "kv_gb", "total_gb", "tokens_per_sec"} <= set(q)
    assert q["verdict"] == "fits fully"
    assert payload["models"][0]["best_quant_on_gpu"]


def test_json_reports_layers_for_the_overspill_plan(capsys):
    _, out, _ = run_cli(
        capsys, *BASE, "--model", "llama-3.3-70b", "--quant", "q4_k_m", "--json"
    )
    q = json.loads(out)["models"][0]["quants"][0]
    assert q["total_layers"] == 80
    assert q["gpu_layers"] + q["cpu_layers"] == 80
    assert q["cpu_layers"] > 0


def test_json_marks_moe_models(capsys):
    _, out, _ = run_cli(capsys, *BASE, "--model", "mixtral-8x7b", "--json")
    model = json.loads(out)["models"][0]
    assert model["is_moe"] is True
    assert model["active_params_b"] < model["params_b"]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_unknown_model_fails_with_the_known_list(capsys):
    code, _, err = run_cli(capsys, *BASE, "--model", "nosuchmodel")
    assert code == cli.EXIT_USAGE
    assert "unknown model" in err
    assert "llama-3.1-8b" in err


def test_unknown_quant_fails_with_the_known_list(capsys):
    code, _, err = run_cli(capsys, *BASE, "--model", "llama-3.1-8b", "--quant", "q9_zz")
    assert code == cli.EXIT_USAGE
    assert "unknown quant" in err
    assert "q4_k_m" in err


def test_missing_model_is_a_usage_error(capsys):
    code, _, err = run_cli(capsys, *BASE)
    assert code == cli.EXIT_USAGE
    assert "--model" in err


def test_zero_vram_is_rejected_not_silently_ignored(capsys):
    """`--vram 0` is an explicit invalid request, not an absent flag."""

    code, out, err = run_cli(capsys, "--model", "llama-3.1-8b", "--vram", "0")
    assert code == cli.EXIT_USAGE
    assert "no GPU memory known" in err
    assert out == ""


def test_missing_inventory_file_is_a_usage_error(capsys):
    code, _, err = run_cli(capsys, "--inventory", "/nope/rig.toml", "--model", "mistral-7b")
    assert code == cli.EXIT_USAGE
    assert "not found" in err


def test_bad_inventory_toml_is_a_usage_error(capsys, tmp_path):
    broken = tmp_path / "broken.toml"
    broken.write_text("this is [not valid toml\n")
    code, _, err = run_cli(capsys, "--inventory", str(broken), "--model", "mistral-7b")
    assert code == cli.EXIT_USAGE
    assert "bad inventory" in err


def test_list_prints_models_quants_and_gpus(capsys):
    code, out, _ = run_cli(capsys, "--list")
    assert code == cli.EXIT_OK
    assert "models:" in out and "quants:" in out and "gpus:" in out
    assert "llama-3.3-70b" in out
    assert "q4_k_m" in out
    assert "rtx-4090-24g" in out
