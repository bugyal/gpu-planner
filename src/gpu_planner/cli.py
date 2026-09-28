"""gpu-planner CLI.

    gpu-planner plan --model llama-3.3-70b --quant q4_k_m
    gpu-planner plan --model 70b --inventory my-rig.toml --context 32768
    gpu-planner --gpu rtx-4090-24g --ram 64 --model llama-3.3-70b --rent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .catalog import (
    QUANT_BITS,
    ModelSpec,
    bytes_per_param,
    cloud_rate,
    get_gpu,
    get_model,
    normalize_quant,
)
from .hardware import Inventory, detect
from .vram import (
    break_even,
    estimate_throughput,
    largest_quant_that_fits,
    overspill_plan,
    plan_fit,
)

EXIT_OK = 0
EXIT_USAGE = 4


class UsageError(Exception):
    """Bad input, to be turned into an exit code by ``main``.

    Helpers raise this rather than calling ``sys.exit`` so the whole CLI stays
    callable and assertable from a test.
    """


def _parse_inventory(path: str) -> Inventory:
    """Read a tiny TOML inventory: gpu, vram_gb, bandwidth_gbs, ram_gb, cpu_cores."""

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - guarded by pyproject
        raise UsageError("needs Python 3.11+ (tomllib)") from None

    file_path = Path(path)
    if not file_path.is_file():
        raise UsageError(f"inventory not found: {path}")
    try:
        data = tomllib.loads(file_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise UsageError(f"bad inventory TOML: {exc}") from None

    inv = Inventory(
        gpu=data.get("gpu"),
        vram_gb=float(data.get("vram_gb", 0) or 0),
        bandwidth_gbs=float(data.get("bandwidth_gbs", 0) or 0),
        cpu_cores=int(data.get("cpu_cores", 1) or 1),
        system_ram_gb=float(data.get("ram_gb", 0) or 0),
        unified_memory=bool(data.get("unified_memory", False)),
        os_name=str(data.get("os", "declared")),
        detected=["inventory file"],
    )
    return inv


def _resolve_inventory(args) -> Inventory:
    if args.inventory:
        inv = _parse_inventory(args.inventory)
    else:
        inv = detect()

    # Explicit flags win over both detection and the file. Test for None, not
    # truthiness: `--vram 0` is a real (invalid) request, not an absent one.
    if args.gpu:
        try:
            spec = get_gpu(args.gpu)
        except KeyError as exc:
            raise UsageError(exc.args[0]) from None
        inv.gpu = spec.name
        inv.vram_gb = spec.vram_gb
        inv.bandwidth_gbs = spec.bandwidth_gbs
        inv.unified_memory = "unified" in spec.note
        if "unified" not in spec.note and not inv.system_ram_gb:
            inv.system_ram_gb = 64.0
    if args.vram is not None:
        inv.vram_gb = args.vram
    if args.bandwidth is not None:
        inv.bandwidth_gbs = args.bandwidth
    if args.ram is not None:
        inv.system_ram_gb = args.ram
    if args.cpu is not None:
        inv.cpu_cores = args.cpu

    if inv.vram_gb <= 0:
        raise UsageError(
            "no GPU memory known. Pass --gpu <name>, --vram <GB>, or --inventory <file>."
        )
    if inv.bandwidth_gbs <= 0:
        # Without bandwidth the throughput column would be fiction.
        inv.bandwidth_gbs = inv.vram_gb * 30
        inv.missed.append("bandwidth (estimated from VRAM — pass --bandwidth for accuracy)")
    return inv


def _render(inv: Inventory, models: list[ModelSpec], quants: list[str], args) -> str:
    out: list[str] = []
    out.append("inventory:")
    out.extend("  " + line for line in inv.summary_lines())
    out.append("")

    for model in models:
        header = f"{model.name}  ({model.params:g}B total"
        if model.is_moe:
            header += f", {model.effective_params:g}B active"
        header += f", {model.layers} layers)"
        out.append(header)
        if model.note:
            out.append(f"  note: {model.note}")

        out.append("")
        out.append(
            f"  {'quant':<9} {'weights':>9} {'kv':>8} {'over':>7} {'total':>8} "
            f"{'verdict':<18} {'tok/s':>7}"
        )
        out.append("  " + "-" * 72)

        rows: list[tuple[str, object, object, object]] = []
        for quant in quants:
            fit = plan_fit(
                model,
                quant,
                gpu_vram_gb=inv.vram_gb,
                system_ram_gb=inv.system_ram_gb,
                context=args.context,
            )
            plan = overspill_plan(fit)
            tps = estimate_throughput(
                model,
                quant,
                gpu_bandwidth_gbs=inv.bandwidth_gbs,
                plan=plan,
                unified_memory=inv.unified_memory,
            )
            rows.append((quant, fit, plan, tps))
            out.append(
                f"  {quant:<9} {fit.weights_gb:>8.2f}G {fit.kv_gb:>7.2f}G "
                f"{fit.overhead_gb:>6.2f}G {fit.total_required_gb:>7.2f}G "
                f"{fit.verdict:<18} {tps.effective_tokens_per_sec:>7.1f}"
            )

        best = largest_quant_that_fits(
            model,
            gpu_vram_gb=inv.vram_gb,
            system_ram_gb=inv.system_ram_gb,
            context=args.context,
        )
        out.append("")
        if best:
            out.append(f"  best quant fully on GPU at {args.context} ctx: {best}")
        else:
            out.append(
                f"  nothing fits fully on GPU at {args.context} ctx — see the overspill plan"
            )

        # Most detailed verdict for the first requested quant (the others are
        # already in the table above).
        chosen = rows[0]
        _, fit, plan, tps = chosen
        out.append("")
        out.append(f"  overspill plan ({plan.total_layers} layers):")
        out.append(f"    {plan.describe(model.name)}")
        out.append(
            f"    max context fully on GPU: {fit.max_context_fitting_gpu:,} tokens"
        )
        out.append(
            f"    max context using GPU+RAM: {fit.max_context_fitting_total:,} tokens"
        )
        out.append("")
        out.append(f"  throughput ({chosen[0]}):")
        out.append(f"    {tps.effective_tokens_per_sec:.1f} tok/s — bottleneck: {tps.bottleneck}")
        if tps.note:
            out.append(f"    {tps.note}")

        if args.rent:
            out.extend(_rent_section(model, chosen[0], tps, inv, args))

        out.append("")
        out.append("-" * 74)
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def _rent_section(model, quant, tps, inv, args) -> list[str]:
    out: list[str] = []
    cloud = args.cloud_rate or cloud_rate(args.gpu or "")
    proxy_note = ""
    if not cloud:
        # Consumer cards are not rented by the hour, so there is no rate to look
        # up. Use a datacentre card of the same class as a labelled proxy rather
        # than refusing to answer — the break-even order of magnitude is the
        # point, and the label keeps it honest.
        proxy = _proxy_cloud_rate(inv.vram_gb)
        if proxy:
            cloud, proxy_note = proxy
        else:
            out.append("")
            out.append(
                "  rent-vs-buy: no cloud rate for this card — pass --cloud-rate <USD/h>"
            )
            return out

    price = args.price
    if price is None:
        from .catalog import GPUS

        spec = GPUS.get((args.gpu or "").lower())
        price = spec.price_usd if spec and spec.price_usd else None
    if price is None:
        out.append("")
        out.append("  rent-vs-buy: no purchase price known — pass --price")
        return out

    # Compare on cost per million tokens so the two sides are like for like:
    # renting buys a whole GPU-hour, owning amortises capex over the same hour.
    be = break_even(
        gpu_price=price,
        cloud_hourly=cloud,
        monthly_hours=args.hours,
        amortise_months=args.amortise_months,
        electricity_watts=args.watts,
    )
    out.append("")
    out.append(f"  rent vs buy (own price ${price:,.0f}, cloud ${cloud:.2f}/h):")
    if proxy_note:
        out.append(f"    {proxy_note}")
    out.append(f"    {be.describe()}")
    if tps.effective_tokens_per_sec > 0:
        own_per_mtok = be.own_hourly / (tps.effective_tokens_per_sec * 3600) * 1e6
        rent_per_mtok = cloud / (tps.effective_tokens_per_sec * 3600) * 1e6
        out.append(
            f"    at {tps.effective_tokens_per_sec:.1f} tok/s: "
            f"own ${own_per_mtok:.2f}/Mtok vs rent ${rent_per_mtok:.2f}/Mtok"
        )
        out.append(
            "    (same throughput assumed for both — cloud cards are usually faster, "
            "so treat the rent side as an upper bound)"
        )
    return out


def _proxy_cloud_rate(vram_gb: float) -> tuple[float, str] | None:
    """Nearest datacentre rate by VRAM, with the substitution stated."""

    from .catalog import CLOUD_RATES, get_gpu

    candidates = []
    for name, rate in CLOUD_RATES.items():
        try:
            spec = get_gpu(name)
        except KeyError:
            continue
        candidates.append((abs(spec.vram_gb - vram_gb), name, rate))
    if not candidates:
        return None
    _, name, rate = min(candidates)
    return rate, f"proxy: no consumer cloud rate exists; using {name} at ${rate:.2f}/h"


def _to_json(inv, models, quants, args) -> dict:
    payload: dict = {
        "inventory": {
            "gpu": inv.gpu,
            "vram_gb": inv.vram_gb,
            "bandwidth_gbs": inv.bandwidth_gbs,
            "ram_gb": inv.system_ram_gb,
            "cpu_cores": inv.cpu_cores,
            "unified_memory": inv.unified_memory,
            "detected": inv.detected,
            "not_probed": inv.missed,
        },
        "context": args.context,
        "models": [],
    }
    for model in models:
        entry = {
            "name": model.name,
            "params_b": model.params,
            "active_params_b": model.effective_params,
            "layers": model.layers,
            "is_moe": model.is_moe,
            "quants": [],
        }
        for quant in quants:
            fit = plan_fit(
                model,
                quant,
                gpu_vram_gb=inv.vram_gb,
                system_ram_gb=inv.system_ram_gb,
                context=args.context,
            )
            plan = overspill_plan(fit)
            tps = estimate_throughput(
                model,
                quant,
                gpu_bandwidth_gbs=inv.bandwidth_gbs,
                plan=plan,
                unified_memory=inv.unified_memory,
            )
            entry["quants"].append(
                {
                    "quant": quant,
                    "weights_gb": round(fit.weights_gb, 3),
                    "kv_gb": round(fit.kv_gb, 3),
                    "overhead_gb": round(fit.overhead_gb, 3),
                    "total_gb": round(fit.total_required_gb, 3),
                    "verdict": fit.verdict,
                    "fits_gpu": fit.fits_gpu,
                    "gpu_layers": plan.gpu_layers,
                    "cpu_layers": plan.cpu_layers,
                    "total_layers": plan.total_layers,
                    "tokens_per_sec": round(tps.effective_tokens_per_sec, 2),
                    "bottleneck": tps.bottleneck,
                    "max_context_gpu": fit.max_context_fitting_gpu,
                    "max_context_total": fit.max_context_fitting_total,
                }
            )
        entry["best_quant_on_gpu"] = largest_quant_that_fits(
            model,
            gpu_vram_gb=inv.vram_gb,
            system_ram_gb=inv.system_ram_gb,
            context=args.context,
        )
        payload["models"].append(entry)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-planner",
        description="Can I run model X on my hardware, and when does renting beat buying?",
    )
    parser.add_argument("--version", action="version", version=f"gpu-planner {__version__}")
    parser.add_argument("--model", action="append", default=[], help="model name or size (repeatable)")
    parser.add_argument("--quant", action="append", default=[], help="quantisation (repeatable)")
    parser.add_argument("--context", type=int, default=8192, help="context window in tokens (default 8192)")
    parser.add_argument("--inventory", help="TOML file describing the machine")
    parser.add_argument("--gpu", help="catalogue GPU name, e.g. rtx-4090-24g")
    parser.add_argument("--vram", type=float, help="override GPU memory in GB")
    parser.add_argument("--ram", type=float, help="override system RAM in GB")
    parser.add_argument("--bandwidth", type=float, help="override memory bandwidth in GB/s")
    parser.add_argument("--cpu", type=int, help="override CPU core count")
    parser.add_argument("--rent", action="store_true", help="include the rent-vs-buy sheet")
    parser.add_argument("--price", type=float, help="purchase price in USD for the rent-vs-buy sheet")
    parser.add_argument("--cloud-rate", type=float, help="cloud USD/hour override")
    parser.add_argument("--hours", type=float, default=200.0, help="hours per month you would use it (default 200)")
    parser.add_argument("--amortise-months", type=int, default=36, help="ownership lifetime in months (36)")
    parser.add_argument("--watts", type=float, default=450.0, help="system power draw under load (450)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--list", action="store_true", help="list known models, quants and GPUs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        return _run(args)
    except UsageError as exc:
        print(f"gpu-planner: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _run(args) -> int:
    if args.list:
        from .catalog import GPUS, MODELS

        print("models:")
        for name, spec in sorted(MODELS.items()):
            extra = f"  [{spec.note}]" if spec.note else ""
            print(f"  {name:<16} {spec.params:>7.2f}B  {spec.layers:>3} layers{extra}")
        print("\nquants:", ", ".join(sorted(QUANT_BITS, key=lambda q: -bytes_per_param(q))))
        print("\ngpus:")
        for name, spec in sorted(GPUS.items()):
            price = f"${spec.price_usd:,.0f}" if spec.price_usd else "n/a"
            print(f"  {name:<16} {spec.vram_gb:>6.0f}GB  {spec.bandwidth_gbs:>6.0f} GB/s  {price}")
        return EXIT_OK

    if not args.model:
        raise UsageError("--model is required (or use --list)")

    inv = _resolve_inventory(args)

    models = []
    for name in args.model:
        try:
            models.append(get_model(name))
        except KeyError as exc:
            raise UsageError(exc.args[0]) from None

    quants = [normalize_quant(q) for q in (args.quant or ["q4_k_m"])]
    unknown = [q for q in quants if q not in QUANT_BITS]
    if unknown:
        raise UsageError(
            f"unknown quant(s) {', '.join(unknown)}; known: "
            f"{', '.join(sorted(QUANT_BITS))}"
        )

    if args.json:
        print(json.dumps(_to_json(inv, models, quants, args), indent=2))
    else:
        print(_render(inv, models, quants, args), end="")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
