"""How much CPU this process may use, and keeping PyTorch's thread count within it.

A container sees every core of the machine it runs on, even when its CPU time is capped to one or
two of them. PyTorch and Ultralytics size their thread pools from the machine's cores, and on a
capped host such as Streamlit Community Cloud those extra threads use up the allowance waiting on
each other, which can make inference many times slower. Nothing here imports PyTorch at module
level, so it is safe to use before PyTorch loads.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
# Read by PyTorch's OpenMP and BLAS libraries when they load.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _read_numbers(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError):
        return None


def cgroup_cpu_limit(root: Path = CGROUP_ROOT) -> float | None:
    """Return the container's CPU cap in cores, or None if it has none or this is not Linux."""
    cgroup_v2 = _read_numbers(root / "cpu.max")
    if cgroup_v2 is not None:
        try:
            if len(cgroup_v2) >= 2 and cgroup_v2[0] != "max" and int(cgroup_v2[1]) > 0:
                return int(cgroup_v2[0]) / int(cgroup_v2[1])
        except ValueError:
            pass
        return None
    quota = _read_numbers(root / "cpu" / "cpu.cfs_quota_us")
    period = _read_numbers(root / "cpu" / "cpu.cfs_period_us")
    try:
        if quota and period and int(quota[0]) > 0 and int(period[0]) > 0:
            return int(quota[0]) / int(period[0])
    except ValueError:
        pass
    return None


def available_cpus(root: Path = CGROUP_ROOT) -> int:
    """Return how many cores this process can really use, counting container caps."""
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count() or 1
    limit = cgroup_cpu_limit(root)
    if limit is not None:
        count = min(count, math.ceil(limit))
    return max(1, count)


def limit_torch_threads(maximum: int | None = None) -> int:
    """Lower PyTorch's thread count to ``maximum``, or to the available cores, and return it.

    It never raises the count. Ultralytics sets up to 8 threads when a model first runs on the
    CPU, so call this after that.
    """
    import torch

    current = int(torch.get_num_threads())
    target = max(1, min(current, maximum or available_cpus()))
    if target != current:
        torch.set_num_threads(target)
    return target
