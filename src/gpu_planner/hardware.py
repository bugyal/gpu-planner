"""Local hardware detection.

Best effort, never fatal: whatever cannot be detected is simply omitted and the
user is told, because a planner that refuses to run without a GPU probe is
useless on a laptop that has none.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field

GIB = 1024**3


@dataclass
class Inventory:
    gpu: str | None = None
    vram_gb: float = 0.0
    bandwidth_gbs: float = 0.0
    cpu_cores: int = 1
    system_ram_gb: float = 0.0
    unified_memory: bool = False
    os_name: str = ""
    detected: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [
            f"os:        {self.os_name}",
            f"cpu:       {self.cpu_cores} cores",
            f"ram:       {self.system_ram_gb:.0f}GB",
        ]
        if self.gpu:
            kind = "unified memory" if self.unified_memory else "discrete"
            lines.append(f"gpu:       {self.gpu} ({self.vram_gb:g}GB, {kind})")
            lines.append(f"bandwidth: {self.bandwidth_gbs:g} GB/s")
        else:
            lines.append("gpu:       none detected — CPU-only")
        if self.missed:
            lines.append(f"not probed: {', '.join(self.missed)}")
        return lines


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def _system_ram_gb() -> float:
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out.strip().isdigit():
            return int(out.strip()) / GIB
    try:
        return _sysconf_ram()
    except Exception:
        return 0.0


def _sysconf_ram() -> float:
    import os

    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return pages * page_size / GIB


def detect(reference: str = "") -> Inventory:
    """Probe the machine. ``reference`` is optional context for the report."""

    inv = Inventory(os_name=f"{platform.system()} {platform.release()}")
    inv.cpu_cores = _cpu_cores()
    inv.system_ram_gb = round(_system_ram_gb(), 1)

    if platform.system() == "Darwin":
        _detect_apple(inv)
    else:
        _detect_nvidia(inv)
        if not inv.gpu:
            _detect_rocm(inv)
    if not inv.gpu:
        inv.missed.append("gpu (no nvidia-smi/rocm-smi, not Apple Silicon)")
    return inv


def _cpu_cores() -> int:
    import os

    return os.cpu_count() or 1


def _detect_apple(inv: Inventory) -> None:
    """Apple Silicon: unified memory, so VRAM is a fraction of system RAM."""

    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    if not chip:
        inv.missed.append("apple chip name")
        return
    # Only Apple Silicon shares memory with the GPU.
    if "Apple" not in chip:
        inv.missed.append("apple gpu (intel mac)")
        return

    inv.unified_memory = True
    inv.cpu_cores = int(_run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip() or inv.cpu_cores)

    from .catalog import GPUS

    # Match the chip against known unified-memory parts; otherwise use the
    # documented default of ~75% of RAM available to the GPU.
    key = chip.split()[1].lower() if len(chip.split()) > 1 else ""
    candidate = None
    for name, spec in GPUS.items():
        if key and key.replace("-", "") in name.replace("-", ""):
            candidate = spec
            break
    if candidate:
        inv.gpu = candidate.name
        inv.bandwidth_gbs = candidate.bandwidth_gbs
        inv.vram_gb = round(inv.system_ram_gb * 0.75, 1)
    else:
        inv.gpu = f"{chip} (unified)"
        inv.bandwidth_gbs = 0.0
        inv.vram_gb = round(inv.system_ram_gb * 0.75, 1)
        inv.missed.append(f"memory bandwidth for {chip}")


def _detect_nvidia(inv: Inventory) -> None:
    if not shutil.which("nvidia-smi"):
        return
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out.strip():
        inv.missed.append("nvidia-smi returned nothing")
        return
    # Take the first GPU; multi-GPU planning is out of scope for v0.1.
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        inv.missed.append("nvidia-smi output shape")
        return
    inv.gpu = parts[0]
    inv.vram_gb = round(float(parts[1]) / 1024.0, 1)

    # Match against the catalogue for bandwidth; otherwise estimate.
    from .catalog import GPUS

    normalized = inv.gpu.lower().replace("nvidia", "").strip().replace(" ", "-")
    for name, spec in GPUS.items():
        if name.split("-")[0] in normalized or normalized.startswith(name.split("-")[0]):
            inv.bandwidth_gbs = spec.bandwidth_gbs
            break
    if not inv.bandwidth_gbs:
        # Fall back on VRAM: bandwidth scales roughly with the memory bus, and
        # this is flagged as an estimate rather than presented as measured.
        inv.bandwidth_gbs = round(inv.vram_gb * 30, 0)
        inv.missed.append("exact bandwidth (estimated from VRAM)")


def _detect_rocm(inv: Inventory) -> None:
    if not shutil.which("rocm-smi"):
        return
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    if "VRAM" not in out:
        inv.missed.append("rocm-smi output shape")
        return
    inv.gpu = "AMD ROCm GPU"
    inv.missed.append("AMD card name and bandwidth")
