"""CPU limits from container settings, and PyTorch's thread count."""

from pathlib import Path

import pytest

from vrpr import cpu


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def test_cgroup_v2_quota_in_cores(tmp_path: Path) -> None:
    write(tmp_path / "cpu.max", "200000 100000\n")
    assert cpu.cgroup_cpu_limit(tmp_path) == 2.0


def test_cgroup_v2_without_a_quota(tmp_path: Path) -> None:
    write(tmp_path / "cpu.max", "max 100000\n")
    assert cpu.cgroup_cpu_limit(tmp_path) is None


def test_cgroup_v1_quota_in_cores(tmp_path: Path) -> None:
    write(tmp_path / "cpu" / "cpu.cfs_quota_us", "50000\n")
    write(tmp_path / "cpu" / "cpu.cfs_period_us", "100000\n")
    assert cpu.cgroup_cpu_limit(tmp_path) == 0.5


def test_cgroup_v1_without_a_quota(tmp_path: Path) -> None:
    write(tmp_path / "cpu" / "cpu.cfs_quota_us", "-1\n")
    write(tmp_path / "cpu" / "cpu.cfs_period_us", "100000\n")
    assert cpu.cgroup_cpu_limit(tmp_path) is None


@pytest.mark.parametrize("text", ["", "garbage", "100 0", "abc 100000"])
def test_unreadable_cgroup_files_mean_no_limit(tmp_path: Path, text: str) -> None:
    write(tmp_path / "cpu.max", text)
    assert cpu.cgroup_cpu_limit(tmp_path) is None


def test_no_cgroup_files_mean_no_limit(tmp_path: Path) -> None:
    assert cpu.cgroup_cpu_limit(tmp_path) is None


def test_available_cpus_rounds_a_fractional_quota_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: 64)
    monkeypatch.delattr(cpu.os, "sched_getaffinity", raising=False)
    write(tmp_path / "cpu.max", "150000 100000\n")
    assert cpu.available_cpus(tmp_path) == 2
    write(tmp_path / "cpu.max", "7800 100000\n")
    assert cpu.available_cpus(tmp_path) == 1


def test_available_cpus_without_a_quota_uses_the_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: 12)
    monkeypatch.delattr(cpu.os, "sched_getaffinity", raising=False)
    assert cpu.available_cpus(tmp_path) == 12
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: None)
    assert cpu.available_cpus(tmp_path) == 1


def test_limit_torch_threads_only_lowers() -> None:
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        assert cpu.limit_torch_threads(2) == 2
        assert torch.get_num_threads() == 2
        assert cpu.limit_torch_threads(8) == 2
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
